import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

from data_fetcher import SimpleDataFetcher
from agents import DataAgent, SentimentAgent, RiskAgent  # Updated to use LM Studio agents

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI-Powered Financial Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar-content {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .lm-studio-status {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .status-connected {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-disconnected {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 30
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None
    if 'lm_studio_status' not in st.session_state:
        st.session_state.lm_studio_status = None

def check_lm_studio_connection():
    """Check if LM Studio API is accessible"""
    try:
        import requests
        lm_studio_endpoint = os.getenv('LM_STUDIO_ENDPOINT', 'http://localhost:1234')
        response = requests.get(f"{lm_studio_endpoint}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False

def create_candlestick_chart(data, symbol):
    """Create candlestick chart with volume"""
    if data is None or len(data) == 0:
        return None
    
    # Create subplots with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} - Price Chart', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Volume chart
    colors = ['red' if close < open else 'green' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Stock Analysis",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False,
        template="plotly_white"
    )
    
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def display_metrics(current_price, company_name, market_cap, pe_ratio):
    """Display key metrics in a formatted layout"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}" if current_price else "N/A"
        )
    
    with col2:
        st.metric(
            label="Company",
            value=company_name if company_name else "N/A"
        )
    
    with col3:
        st.metric(
            label="Market Cap",
            value=f"${market_cap/1e9:.2f}B" if market_cap else "N/A"
        )
    
    with col4:
        st.metric(
            label="P/E Ratio",
            value=f"{pe_ratio:.2f}" if pe_ratio else "N/A"
        )

def display_lm_studio_status():
    """Display LM Studio connection status"""
    lm_studio_connected = check_lm_studio_connection()
    st.session_state.lm_studio_status = lm_studio_connected
    
    if lm_studio_connected:
        st.sidebar.markdown(
            '<div class="lm-studio-status status-connected">🟢 LM Studio: Connected</div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            '<div class="lm-studio-status status-disconnected">🔴 LM Studio: Disconnected</div>',
            unsafe_allow_html=True
        )
        st.sidebar.warning("⚠️ LM Studio not detected. AI analysis will use fallback responses.")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Main title
    st.markdown('<h1 class="main-header">AI-Powered Financial Intelligence Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.sidebar.header("📊 Dashboard Controls")
    
    # LM Studio status
    display_lm_studio_status()
    
    # Stock symbol input
    symbol = st.sidebar.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter a stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
    ).upper()
    
    # Auto-refresh controls
    st.sidebar.subheader("🔄 Auto Refresh")
    auto_refresh = st.sidebar.toggle("Enable Auto Refresh", value=st.session_state.auto_refresh)
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=10,
            max_value=300,
            value=st.session_state.refresh_interval,
            step=10
        )
        st.session_state.refresh_interval = refresh_interval
    
    st.session_state.auto_refresh = auto_refresh
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now", type="primary"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # LM Studio configuration info
    if st.sidebar.expander("🤖 LM Studio Info"):
        st.write("**Model Path:**")
        st.code("TheBloke\\Llama-2-7B-Chat-GGUF")
        st.write("**Endpoint:** http://localhost:1")
        st.write("**Status:** " + ("✅ Connected" if st.session_state.lm_studio_status else "❌ Disconnected"))
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Display last update time
    if st.session_state.last_refresh:
        st.sidebar.info(f"Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    # Initialize components
    data_fetcher = SimpleDataFetcher()
    data_agent = DataAgent()
    sentiment_agent = SentimentAgent()
    risk_agent = RiskAgent()
    
    # Main content area
    if symbol:
        try:
            # Fetch data
            with st.spinner(f"📡 Fetching data for {symbol}..."):
                stock_data, current_price, company_name, market_cap, pe_ratio = data_fetcher.get_stock_data(symbol)
                news_data = data_fetcher.get_simple_news(symbol)
            
            if stock_data is not None and not stock_data.empty:
                # Create tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Charts", "📰 Sentiment", "⚠️ Risk"])
                
                with tab1:
                    st.subheader(f"📈 {symbol} Overview")
                    
                    # Display key metrics
                    display_metrics(current_price, company_name, market_cap, pe_ratio)
                    
                    # AI Analysis
                    st.subheader("🤖 AI Analysis")
                    with st.spinner("Generating AI insights using LM Studio..."):
                        try:
                            ai_analysis = data_agent.analyze_stock(symbol, current_price, market_cap, pe_ratio)
                            if st.session_state.lm_studio_status:
                                st.success("🤖 **LM Studio Analysis:**")
                            else:
                                st.info("📊 **Fallback Analysis:**")
                            st.write(ai_analysis)
                        except Exception as e:
                            st.warning(f"AI analysis error: {str(e)}")
                            st.info("📊 Based on current metrics, this appears to be a stable stock with regular trading activity. Monitor for any significant changes in volume or price patterns.")
                
                with tab2:
                    st.subheader(f"📊 {symbol} Price Charts")
                    
                    # Create and display charts
                    fig = create_candlestick_chart(stock_data, symbol)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Unable to create chart. Please check the stock symbol.")
                
                with tab3:
                    st.subheader(f"📰 {symbol} News Sentiment")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Recent News Headlines:**")
                        for i, article in enumerate(news_data[:5], 1):
                            st.write(f"{i}. {article['title']}")
                        
                        # LLM Sentiment Analysis
                        if st.session_state.lm_studio_status:
                            with st.expander("🤖 AI Sentiment Analysis"):
                                with st.spinner("Analyzing news sentiment with LM Studio..."):
                                    try:
                                        llm_sentiment = sentiment_agent.analyze_news_sentiment_llm(news_data, symbol)
                                        st.write(llm_sentiment)
                                    except Exception as e:
                                        st.write("LLM sentiment analysis unavailable.")
                    
                    with col2:
                        # Calculate sentiment
                        sentiment_score = sentiment_agent.analyze_sentiment(news_data)
                        
                        # Display sentiment gauge
                        fig_sentiment = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=sentiment_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Sentiment Score"},
                            delta={'reference': 0},
                            gauge={
                                'axis': {'range': [-1, 1]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [-1, -0.3], 'color': "red"},
                                    {'range': [-0.3, 0.3], 'color': "yellow"},
                                    {'range': [0.3, 1], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 0
                                }
                            }
                        ))
                        fig_sentiment.update_layout(height=300)
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        # Sentiment interpretation
                        if sentiment_score > 0.1:
                            st.success("📈 Positive sentiment")
                        elif sentiment_score < -0.1:
                            st.error("📉 Negative sentiment")
                        else:
                            st.warning("📊 Neutral sentiment")
                
                with tab4:
                    st.subheader(f"⚠️ {symbol} Risk Assessment")
                    
                    # Calculate risk metrics
                    volatility, risk_level = risk_agent.calculate_risk(stock_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="30-Day Volatility",
                            value=f"{volatility:.4f}" if volatility else "N/A"
                        )
                        
                        # Risk level indicator
                        if risk_level == "Low":
                            st.success(f"🟢 Risk Level: {risk_level}")
                        elif risk_level == "Medium":
                            st.warning(f"🟡 Risk Level: {risk_level}")
                        else:
                            st.error(f"🔴 Risk Level: {risk_level}")
                        
                        # LLM Risk Analysis
                        if st.session_state.lm_studio_status and volatility:
                            with st.expander("🤖 AI Risk Analysis"):
                                with st.spinner("Analyzing risk with LM Studio..."):
                                    try:
                                        risk_analysis = risk_agent.analyze_risk_llm(volatility, risk_level, symbol)
                                        st.write(risk_analysis)
                                    except Exception as e:
                                        st.write("LLM risk analysis unavailable.")
                    
                    with col2:
                        # Create volatility chart
                        if stock_data is not None and len(stock_data) > 30:
                            returns = stock_data['Close'].pct_change().dropna()
                            rolling_vol = returns.rolling(window=30).std()
                            
                            fig_vol = go.Figure()
                            fig_vol.add_trace(go.Scatter(
                                x=rolling_vol.index,
                                y=rolling_vol,
                                mode='lines',
                                name='30-Day Rolling Volatility',
                                line=dict(color='red', width=2)
                            ))
                            
                            fig_vol.update_layout(
                                title="30-Day Rolling Volatility",
                                xaxis_title="Date",
                                yaxis_title="Volatility",
                                height=300,
                                template="plotly_white"
                            )
                            
                            st.plotly_chart(fig_vol, use_container_width=True)
                    
                    # Risk explanation
                    st.info("""
                    **Risk Assessment Explanation:**
                    - **Low Risk** (Volatility < 0.02): Stable stock with minimal price fluctuations
                    - **Medium Risk** (0.02 ≤ Volatility < 0.04): Moderate price movements expected
                    - **High Risk** (Volatility ≥ 0.04): High volatility with significant price swings
                    """)
            
            else:
                st.error(f"❌ Unable to fetch data for {symbol}. Please check the symbol and try again.")
                st.info("💡 **Tip:** Make sure you're using a valid stock ticker symbol (e.g., AAPL, TSLA, MSFT)")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please try again or contact support if the issue persists.")
    
    else:
        st.info("👆 Please enter a stock symbol in the sidebar to get started!")
        st.markdown("""
        ### 🚀 Getting Started with LM Studio
        
        1. **Start LM Studio** with your local model
        2. **Enable API Server** on port 1234
        3. **Load your model:** `TheBloke/Llama-2-7B-Chat-GGUF`
        4. **Enter a stock symbol** in the sidebar
        5. **Enjoy AI-powered financial analysis!**
        """)
    
    # Auto refresh logic
    if auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
