import streamlit as st
import streamlit.components.v1 as components

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
def fetch_binance_data(symbol='BTC-USD', timeframe='1h', years=2):
    """
    Fetches historical OHLCV data from Yahoo Finance (yfinance).
    """
    import yfinance as yf
    import time
    
    status_text = st.empty()
    status_text.text("Fetching data from Yahoo Finance...")
    
    # Retry logic for rate limiting
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Yahoo Finance has limits on hourly data (max 730 days)
            # Use shorter period to reduce API load
            if timeframe == '1h':
                # Reduced to 365 days to minimize rate limit issues
                period = '365d'
                if attempt == 0:  # Only show warning on first attempt
                    st.info("📊 Fetching 1 year of hourly Bitcoin data from Yahoo Finance...")
            else:
                period = f'{years}y'
            
            # Download with period parameter
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=timeframe)
            
            if df.empty:
                if attempt < max_retries - 1:
                    status_text.text(f"Retry {attempt + 1}/{max_retries}... waiting {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    st.error(f"❌ No data found for {symbol} after {max_retries} attempts. Yahoo Finance may be rate limiting.")
                    status_text.empty()
                    return pd.DataFrame()
                
            # Standardize columns to lowercase for compatibility
            df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Remove timezone if present
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            status_text.empty()
            st.success(f"✅ Loaded {len(df):,} candles from Yahoo Finance")
            return df
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate' in error_msg or '429' in error_msg or 'too many' in error_msg:
                if attempt < max_retries - 1:
                    status_text.text(f"⏳ Rate limited. Retry {attempt + 1}/{max_retries} in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    status_text.empty()
                    st.error("⚠️ Yahoo Finance rate limit reached. Please wait a few minutes and refresh the page.")
                    return pd.DataFrame()
            else:
                status_text.empty()
                st.error(f"❌ Error fetching data: {str(e)}")
                return pd.DataFrame()
    
    status_text.empty()
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

def find_similar_patterns(df, current_window_size=50, top_k=None):
    """
    The Core Engine:
    1. Extracts the current market context (Regime + Volatility).
    2. Filters history for ONLY that context.
    3. Uses NearestNeighbors on Z-Score normalized vectors to find matches.
    
    If top_k is None, returns ALL matches. Otherwise returns top K nearest.
    """
    
    # Safety check
    if len(df) < current_window_size + 200:
        return None, None, "Not enough data."

    # Get Current Context
    current_data = df.iloc[-current_window_size:]
    current_regime = current_data['regime'].iloc[-1] # Regime of the *latest* candle
    current_vol_regime = current_data['vol_regime'].iloc[-1] # Volatility regime
    
    current_prices = current_data['close'].values
    current_vector = normalize_window(current_prices).reshape(1, -1)
    
    # Filter History: Create a candidate set
    # We want subsequences where the END matches BOTH regime AND volatility context
    # We must exclude the most recent window itself from the search "history"
    
    # 1. Identify valid end indices in history
    # Valid index 'i' means df.iloc[i] is the *last* candle of a candidate window
    # We define history as everything up to 'current_window_size' candles ago to avoid overlap
    history_cutoff_idx = len(df) - current_window_size - 1
    
    # Create the search space
    # It constructs a sliding window matrix. This can be memory intensive for huge data, 
    # but for 50k rows it's manageable (50k * 50 floats).
    
    # Optimization: Filter indices first by BOTH regime AND volatility to reduce matrix creation size
    # Candidates are indices i where:
    # - df['regime'].iloc[i] == current_regime (BULL/BEAR match)
    # - df['vol_regime'].iloc[i] == current_vol_regime (HIGH_VOL/LOW_VOL match)
    # - i >= window_size (to have enough start data)
    # - i < history_cutoff_idx
    
    valid_indices = np.where(
        (df['regime'] == current_regime) & 
        (df['vol_regime'] == current_vol_regime) &
        (np.arange(len(df)) >= current_window_size) & 
        (np.arange(len(df)) < history_cutoff_idx)
    )[0]
    
    if len(valid_indices) < 1:
        return None, None, f"Not enough historical {current_regime} + {current_vol_regime} patterns found."
    
    # Determine how many neighbors to find
    if top_k is None:
        # Find ALL matches
        n_neighbors = len(valid_indices)
    else:
        # Find top K matches
        n_neighbors = min(top_k, len(valid_indices))

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
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
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
        
        # Calculate percentage changes at different intervals
        start_price = future_prices[0]
        pct_changes = {}
        for hours in [1, 2, 3, 6, 12, 24]:
            if hours < len(future_prices):
                pct_changes[f'{hours}h'] = ((future_prices[hours] - start_price) / start_price) * 100
        
        matches.append({
            'timestamp': timestamp,
            'distance': distance,
            'window_prices': history_window_prices,
            'future_prices': future_prices,
            'regime': df['regime'].iloc[original_idx],
            'return': (future_prices[-1] - future_prices[0]) / future_prices[0],
            'pct_changes': pct_changes,
            'start_price': start_price,
            'end_price': future_prices[-1]
        })
    
    # De-duplicate overlapping matches
    # Filter out matches that are within 24 hours of each other (keep the most similar one)
    if len(matches) > 1:
        deduplicated_matches = []
        used_timestamps = set()
        
        for match in matches:
            timestamp = match['timestamp']
            
            # Check if this timestamp is too close to any already selected match
            is_duplicate = False
            for used_ts in used_timestamps:
                time_diff = abs((timestamp - used_ts).total_seconds() / 3600)  # Hours
                if time_diff < 100:  # Within 100 hours
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated_matches.append(match)
                used_timestamps.add(timestamp)
        
        matches = deduplicated_matches
        
    return current_prices, matches, None

def render_lightweight_charts(current_prices, matches_to_plot, window_size):
    """
    Renders Lightweight Charts with current price and ghost lines.
    Returns HTML string for st.components.html
    """
    import json
    
    # Prepare current price data
    current_data = [{"time": i, "value": float(price)} for i, price in enumerate(current_prices)]
    
    # Prepare ghost lines data
    ghost_lines = []
    for idx, m in enumerate(matches_to_plot):
        # Reconstruct the ghost line (simplified - just use raw prices for now)
        window_prices = m['window_prices']
        future_prices = m['future_prices']
        full_prices = np.concatenate([window_prices, future_prices])
        
        # Create data points
        line_data = [{"time": i, "value": float(price)} for i, price in enumerate(full_prices)]
        
        # Determine color based on outcome
        color = '#00ff00' if m['return'] > 0 else '#ff0000'
        
        ghost_lines.append({
            'data': line_data,
            'color': color,
            'label': m['timestamp'].strftime('%Y-%m-%d')
        })
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; }}
            #chart {{ width: 100%; height: 600px; }}
        </style>
    </head>
    <body>
        <div id="chart"></div>
        <script>
            const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
                width: window.innerWidth,
                height: 600,
                layout: {{
                    background: {{ color: '#ffffff' }},
                    textColor: '#333',
                }},
                grid: {{
                    vertLines: {{ color: '#f0f0f0' }},
                    horzLines: {{ color: '#f0f0f0' }},
                }},
            }});

            // Add current price line (main)
            const currentSeries = chart.addLineSeries({{
                color: '#000000',
                lineWidth: 3,
                title: 'Current Price'
            }});
            currentSeries.setData({json.dumps(current_data)});

            // Add ghost lines
            const ghostData = {json.dumps(ghost_lines)};
            ghostData.forEach((ghost, index) => {{
                const series = chart.addLineSeries({{
                    color: ghost.color,
                    lineWidth: 1,
                    title: ghost.label,
                    priceLineVisible: false,
                    lastValueVisible: false
                }});
                series.setData(ghost.data);
            }});

            // Add vertical line at "Now"
            const nowMarker = {{
                time: {window_size - 1},
                position: 'inBar',
                color: '#888',
                shape: 'arrowDown',
                text: 'Now'
            }};

            chart.timeScale().fitContent();
        </script>
    </body>
    </html>
    """
    
    return html_template

# -----------------------------------------------------------------------------
# 3. Streamlit UI
# -----------------------------------------------------------------------------

# Sidebar Controls
window_size = st.sidebar.slider("Pattern Window Size", 20, 100, 50)
enable_glow = st.sidebar.checkbox("✨ Enable Ghost Line Glow", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Match Settings:**")
show_all_matches = st.sidebar.checkbox("📊 Show ALL Matches", value=False, help="Find all historical matches instead of just top 5")

if not show_all_matches:
    num_matches = st.sidebar.slider("Number of Matches", 3, 20, 5)
else:
    num_matches = None
    st.sidebar.info("Will find ALL matching patterns")

st.sidebar.markdown("---")
st.sidebar.markdown("**Chart Settings:**")
use_lightweight_charts = st.sidebar.checkbox("📈 Use Lightweight Charts", value=False, help="Use TradingView's Lightweight Charts (shows top 20, better performance)")

st.sidebar.markdown("---")
st.sidebar.text("Settings:")
st.sidebar.text(f"Symbol: BTC-USD")
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
    last_vol_regime = df_processed['vol_regime'].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${last_price:,.2f}")
    col2.metric("Market Regime", last_regime, delta="Bullish" if last_regime=="BULL" else "-Bearish")
    col3.metric("Volatility", last_vol_regime, delta="High" if last_vol_regime=="HIGH_VOL" else "Low")
    col4.metric("Available History", f"{len(df_processed)} hours")
    
    # Run Search
    st.subheader(f"Current Pattern Search (Last {window_size} Candles)")
    
    current_prices, matches, error_msg = find_similar_patterns(df_processed, current_window_size=window_size, top_k=num_matches)
    
    if error_msg:
        st.warning(error_msg)
    else:
        # Performance optimization: Adjust matches to plot based on chart type
        if use_lightweight_charts:
            # Lightweight Charts can handle more matches efficiently
            if show_all_matches and len(matches) > 20:
                st.info(f"📊 Found {len(matches)} total matches (after de-duplication). Displaying top 20 most similar on Lightweight Chart, but statistics include ALL matches.")
                matches_to_plot = matches[:20]  # Plot top 20 with Lightweight Charts
                all_matches = matches
            else:
                matches_to_plot = matches
                all_matches = matches
        else:
            # Plotly - limit to 10 for performance
            if show_all_matches and len(matches) > 10:
                st.info(f"📊 Found {len(matches)} total matches (after de-duplication). Displaying top 10 most similar on chart, but statistics include ALL matches.")
                matches_to_plot = matches[:10]  # Only plot top 10
                all_matches = matches  # Keep all for statistics
            else:
                matches_to_plot = matches
                all_matches = matches
        
        # ---------------------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------------------
        if use_lightweight_charts:
            # Use Lightweight Charts
            st.subheader("📈 Lightweight Charts View")
            html_content = render_lightweight_charts(current_prices, matches_to_plot, window_size)
            components.html(html_content, height=650)
        else:
            # Use Plotly (existing code)
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
            
            # Calculate statistics on ALL matches
            for m in all_matches:
                if m['return'] > 0:
                    up_matches += 1
                else:
                    down_matches += 1
            
            # But only plot the subset
            for i, m in enumerate(matches_to_plot):
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
            
            # Apply glow effect if enabled
            line_config = {'color': color, 'width': 2 if enable_glow else 1}
            
            fig.add_trace(go.Scatter(
                x=full_x,
                y=full_y,
                mode='lines',
                name=f'Match: {date_str}',
                line=line_config,
                hoverinfo='name'
            ))
            
            # Add glow effect (shadow lines)
            if enable_glow:
                # Reduce glow layers when plotting many matches
                glow_layers = [4, 6] if len(matches_to_plot) > 5 else [4, 6, 8]
                for glow_width in glow_layers:
                    glow_opacity = 0.1 if outcome_return > 0 else 0.08
                    glow_color = f'rgba(0, 255, 0, {glow_opacity})' if outcome_return > 0 else f'rgba(255, 0, 0, {glow_opacity})'
                    fig.add_trace(go.Scatter(
                        x=full_x,
                        y=full_y,
                        mode='lines',
                        line=dict(color=glow_color, width=glow_width),
                        showlegend=False,
                        hoverinfo='skip'
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
        total_matches = len(all_matches)  # Use all_matches for statistics
        win_rate = (up_matches / total_matches * 100) if total_matches > 0 else 0
        
        st.markdown(f"""
        ### 📊 Analysis Summary
        Found **{total_matches}** historically similar patterns within **{last_regime} + {last_vol_regime}** context.
        """)
        
        # Display comprehensive statistics
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Total Matches", total_matches)
        with stat_col2:
            st.metric("UP Scenarios 📈", up_matches, delta=f"{up_matches/total_matches*100:.1f}%")
        with stat_col3:
            st.metric("DOWN Scenarios 📉", down_matches, delta=f"-{down_matches/total_matches*100:.1f}%")
        with stat_col4:
            st.metric("Win Rate", f"{win_rate:.1f}%", delta="Bullish" if win_rate > 50 else "Bearish")
        
        
        # Detailed Match Statistics
        st.markdown("### 🔍 Detailed Match Statistics")
        
        # Calculate average percentage changes across all matches
        avg_changes = {}
        for hours in [1, 2, 3, 6, 12, 24]:
            hour_key = f'{hours}h'
            changes = [m['pct_changes'].get(hour_key, 0) for m in all_matches if hour_key in m['pct_changes']]
            if changes:
                avg_changes[hour_key] = np.mean(changes)
        
        # Display average changes
        if avg_changes:
            cols = st.columns(len(avg_changes))
            for idx, (period, avg_pct) in enumerate(avg_changes.items()):
                with cols[idx]:
                    delta_color = "normal" if avg_pct >= 0 else "inverse"
                    st.metric(
                        label=f"Avg {period}",
                        value=f"{avg_pct:+.2f}%",
                        delta=f"{'↑' if avg_pct > 0 else '↓'}"
                    )
        
        # Individual match details - limit display to prevent UI overload
        st.markdown("### 📋 Individual Match Details")
        
        # Show only first 20 matches in detail to prevent UI lag
        matches_to_show = all_matches[:20] if len(all_matches) > 20 else all_matches
        if len(all_matches) > 20:
            st.info(f"Showing detailed info for top 20 matches out of {len(all_matches)} total.")
        
        for idx, m in enumerate(matches_to_show, 1):
            with st.expander(f"Match #{idx}: {m['timestamp'].strftime('%Y-%m-%d %H:%M')} ({'UP' if m['return'] > 0 else 'DOWN'})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Pattern Info:**")
                    st.write(f"- **Date:** {m['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"- **Regime:** {m['regime']}")
                    st.write(f"- **Similarity Score:** {m['distance']:.4f}")
                    st.write(f"- **Overall Return:** {m['return']*100:+.2f}%")
                
                with col2:
                    st.markdown("**Hourly Price Changes:**")
                    for period, pct in m['pct_changes'].items():
                        emoji = "🟢" if pct > 0 else "🔴"
                        st.write(f"{emoji} **{period}:** {pct:+.2f}%")
        
        st.markdown("""
        ---
        *Note: The colored lines are historical price actions re-scaled (normalized) to fit the current price level. 
        This shows how market participants reacted to similar structures in the past.*
        """)
