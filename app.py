
app.py


import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Bitcoin Market Memory", layout="wide")
st.title("🧠 Context-Aware Market Memory: BTC/USDT")
st.markdown("""
**Concept:** This tool finds "Ghost Lines" — historical price patterns that match the *current* market structure.
It uses Z-score normalization to match shapes rather than absolute prices and filters by market regime (Bull/Bear).
""")
# -----------------------------------------------------------------------------
# 1. Data Ingestion (with Caching)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600*12)  # Cache for 12 hours
def fetch_binance_data(symbol='BTC-USD', timeframe='1h', years=5):
    """
    Fetches historical OHLCV data from Yahoo Finance (yfinance).
    """
    import yfinance as yf
    
    # Calculate start date
    now = datetime.now()
    start_date = now - timedelta(days=365 * years)
    
    status_text = st.empty()
    status_text.text("Fetching data from Yahoo Finance...")
    
    try:
        # yfinance allows fetching by period/interval easily
        # '1h' interval has a limitation on how far back it goes (730 days max usually for hourly)
        # If years > 2, yfinance might truncate or we might need daily if user wants 5 years of hourly.
        # However, for '1h', yfinance often limits to last 730 days. 
        # Let's try to get max available for 1h.
        
        # Override years if > 2 for hourly data due to YF limitation
        if timeframe == '1h' and years > 2:
            st.warning("Note: Yahoo Finance limits 1h data to the last 730 days (~2 years). Fetching max available.")
        
        # Download
        df = yf.download(symbol, start=start_date, interval=timeframe, progress=False)
        
        if df.empty:
            st.error("No data found for symbol.")
            return pd.DataFrame()
            
        # YFinance returns MultiIndex columns sometimes (Price, Ticker). Fix that.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Standardize columns to lowercase for compatibility
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # Filter timezone if present to avoid mixups
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        status_text.empty()
        return df
        
    except Exception as e:
        status_text.empty()
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
# -----------------------------------------------------------------------------
# 2. Feature Engineering & Logic
# -----------------------------------------------------------------------------
def process_data(df):
    """
    Adds technical indicators and labels for Context-Awareness.
    """
    df = df.copy()
    
    # 2.1 Technical Indicators
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # ATR Calculation
    # TR = Max(High-Low, Abs(High-PrevClose), Abs(Low-PrevClose))
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['ATR_MA'] = df['ATR'].rolling(window=200).mean() # Smoothed avg of volatility for regime check
    # 2.2 Regime Labeling
    # BULL: Price > SMA_200, BEAR: Price < SMA_200
    df['regime'] = np.where(df['close'] > df['SMA_200'], 'BULL', 'BEAR')
    
    # Volatility Labeling
    df['vol_regime'] = np.where(df['ATR'] > df['ATR_MA'], 'HIGH_VOL', 'LOW_VOL')
    
    # Clean NaN values created by rolling windows
    df.dropna(inplace=True)
    return df
def normalize_window(window_prices):
    """
    Z-Score Normalization: (Price - Mean) / StdDev
    Makes the shape comparable regardless of absolute price level.
    """
    return zscore(window_prices)
def find_similar_patterns(df, current_window_size=50, top_k=5):
    """
    The Core Engine:
    1. Extracts the current market context (Regime).
    2. Filters history for ONLY that regime.
    3. Uses NearestNeighbors on Z-Score normalized vectors to find matches.
    """
    
    # Safety check
    if len(df) < current_window_size + 200:
        return None, None, "Not enough data."
    # Get Current Context
    current_data = df.iloc[-current_window_size:]
    current_regime = current_data['regime'].iloc[-1] # Regime of the *latest* candle
    # Note: We could use majority vote for regime, but using latest is stricter/simpler.
    
    current_prices = current_data['close'].values
    current_vector = normalize_window(current_prices).reshape(1, -1)
    
    # Filter History: Create a candidate set
    # We want subsequences of length 50 where the END of the subsequence matches the current regime
    # We must exclude the most recent window itself from the search "history"
    
    # 1. Identify valid end indices in history
    # Valid index 'i' means df.iloc[i] is the *last* candle of a candidate window
    # We define history as everything up to 'current_window_size' candles ago to avoid overlap
    history_cutoff_idx = len(df) - current_window_size - 1
    
    # Create the search space
    # It constructs a sliding window matrix. This can be memory intensive for huge data, 
    # but for 50k rows it's manageable (50k * 50 floats).
    
    # Optimization: Filter indices first by regime to reduce matrix creation size
    # Candidates are indices i where df['regime'].iloc[i] == current_regime
    # AND i > window_size (to have enough start data) AND i < history_cutoff_idx
    
    valid_indices = np.where(
        (df['regime'] == current_regime) & 
        (np.arange(len(df)) >= current_window_size) & 
        (np.arange(len(df)) < history_cutoff_idx)
    )[0]
    
    if len(valid_indices) < top_k:
        return None, None, f"Not enough historical {current_regime} patterns found."
    # Construct feature matrix X
    # Each row is a normalized window of size 50 ending at index i
    X = []
    
    # We store the mapping from matrix index back to dataframe index
    index_mapping = [] 
    
    # To speed up, we can use a loop or rolling.apply. Loop is easier to read for MVP.
    # For 40k candles, this might take a few seconds.
    # Vectorized approach: stride_tricks is faster but complex. Let's stick to list comprehension for clarity unless slow.
    close_prices = df['close'].values
    
    for idx in valid_indices:
        # Window is from (idx - window_size + 1) to idx (inclusive)
        window = close_prices[idx - current_window_size + 1 : idx + 1]
        norm_window = normalize_window(window)
        X.append(norm_window)
        index_mapping.append(idx)
        
    X = np.array(X)
    
    # Fit Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=top_k, metric='euclidean')
    nn.fit(X)
    
    # Search
    distances, neighbor_indices = nn.kneighbors(current_vector)
    
    # Retrieve match details
    matches = []
    for i, neighbor_idx in enumerate(neighbor_indices[0]):
        original_idx = index_mapping[neighbor_idx]
        distance = distances[0][i]
        
        # Look ahead data (Forward projection)
        # e.g., next 20 candles
        look_ahead = 20
        start_idx = original_idx - current_window_size + 1
        end_idx = original_idx
        
        # Check bounds for lookahead
        if original_idx + look_ahead >= len(df):
            continue
            
        history_window_prices = df.iloc[start_idx : end_idx + 1]['close'].values
        future_prices = df.iloc[end_idx + 1 : end_idx + 1 + look_ahead]['close'].values
        timestamp = df.index[original_idx]
        
        matches.append({
            'timestamp': timestamp,
            'distance': distance,
            'window_prices': history_window_prices,
            'future_prices': future_prices,
            'regime': df['regime'].iloc[original_idx],
            'return': (future_prices[-1] - future_prices[0]) / future_prices[0]
        })
        
    return current_prices, matches, None
# -----------------------------------------------------------------------------
# 3. Streamlit UI
# -----------------------------------------------------------------------------
# Sidebar Controls
window_size = st.sidebar.slider("Pattern Window Size", 20, 100, 50)
st.sidebar.markdown("---")
st.sidebar.text("Settings:")
st.sidebar.text(f"Symbol: BTC/USDT")
st.sidebar.text(f"Timeframe: 1h")
# Main execution
with st.spinner("Loading Market Data..."):
    df = fetch_binance_data()
    
if df.empty:
    st.error("Failed to fetch data.")
else:
    df_processed = process_data(df)
    
    # Current Market Info
    last_price = df_processed['close'].iloc[-1]
    last_sma = df_processed['SMA_200'].iloc[-1]
    last_regime = df_processed['regime'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${last_price:,.2f}")
    col2.metric("Market Regime", last_regime, delta="Bullish" if last_regime=="BULL" else "-Bearish")
    col3.metric("Available History", f"{len(df_processed)} hours")
    
    # Run Search
    st.subheader(f"Current Pattern Search (Last {window_size} Candles)")
    
    current_prices, matches, error_msg = find_similar_patterns(df_processed, current_window_size=window_size)
    
    if error_msg:
        st.warning(error_msg)
    else:
        # ---------------------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------------------
        fig = go.Figure()
        
        # 1. Plot Current Pattern (Black Line)
        # We index it 0 to N
        x_current = list(range(len(current_prices)))
        fig.add_trace(go.Scatter(
            x=x_current, 
            y=current_prices, 
            mode='lines', 
            name='Current Price',
            line=dict(color='black', width=3)
        ))
        
        # 2. Plot Ghost Lines
        # We need to scale the matches to overlay them visually on the current price.
        # Since we matched on Z-scores, the absolute levels might differ.
        # Approach: Re-scale the matched window to start at the same price as current window's start,
        # OR just plot them on a secondary axis?
        # Better: "Un-normalize" the ghost line to fit the current range roughly, 
        # or more simply: Align the first point of the match to the first point of current.
        
        # "Shape matching" visualization implies we care about the move, not the level.
        # We will re-base the ghost lines so they start at the same price as the current sequence[0].
        # Even better: Use the normalization parameters (mean/std) of the CURRENT window to 
        # project the z-scores of the match back to current price levels. -> This is the most mathematically correct "Ghost Projection"
        
        current_mean = np.mean(current_prices)
        current_std = np.std(current_prices)
        
        up_matches = 0
        down_matches = 0
        
        for i, m in enumerate(matches):
            # Reconstruct price series from the match's Z-scores (implicitly)
            # Actually we have the raw prices. Let's calculate its Z-score and project to current.
            gw_prices = m['window_prices']
            gw_mean = np.mean(gw_prices)
            gw_std = np.std(gw_prices)
            
            # Z = (P - mu) / sigma  =>  P_projected = Z * current_sigma + current_mu
            gw_z = (gw_prices - gw_mean) / gw_std
            
            # Project History Window
            projected_window = gw_z * current_std + current_mean
            
            # Project Future (Forecast)
            future_prices = m['future_prices']
            # We treat the future as a continuation of the Z-score path
            # But we must use the SAME match stats (gw_mean, gw_std) to convert future prices to Z
            future_z = (future_prices - gw_mean) / gw_std
            projected_future = future_z * current_std + current_mean
            
            # Determine color based on outcome
            outcome_return = m['return']
            if outcome_return > 0:
                color = 'rgba(0, 255, 0, 0.3)' # Greenish transparent
                up_matches += 1
            else:
                color = 'rgba(255, 0, 0, 0.3)' # Reddish transparent
                down_matches += 1
                
            # Plot the combined path (History + Future) as one ghost line
            # x-coordinates: 0 to 50 for window, 50 to 70 for future
            full_y = np.concatenate([projected_window, projected_future])
            full_x = list(range(len(full_y)))
            
            date_str = m['timestamp'].strftime('%Y-%m-%d')
            
            fig.add_trace(go.Scatter(
                x=full_x,
                y=full_y,
                mode='lines',
                name=f'Match: {date_str}',
                line=dict(color=color, width=1),
                hoverinfo='name'
            ))
            # Mark the "Now" point on the ghost line
            fig.add_trace(go.Scatter(
                x=[window_size-1],
                y=[projected_window[-1]],
                mode='markers',
                marker=dict(color=color, size=5),
                showlegend=False
            ))
        # Divider line for "Now"
        fig.add_vline(x=window_size-1, line_width=1, line_dash="dash", line_color="gray", annotation_text="Now")
        
        fig.update_layout(
            title="Ghost Lines: Historical Context Matches",
            xaxis_title="Hours (0 to 50 = Past, 50+ = Future projection)",
            yaxis_title="Price (Projected)",
            template="plotly_white",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        st.markdown(f"""
        ### Analysis
        Found **{len(matches)}** historically similar patterns within the **{last_regime}** regime.
        - **{up_matches}** scenarios went UP.
        - **{down_matches}** scenarios went DOWN.
        
        *Note: The colored lines are historical price actions re-scaled (normalized) to fit the current price level. 
        This shows how market participants reacted to similar structures in the past.*
        """)
