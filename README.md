# AI-Powered Financial Intelligence Web App - LM Studio Version

A sophisticated Streamlit dashboard that showcases a multi-agent AI system for real-time stock analysis using **your local Llama-2-7B-Chat-GGUF model** via LM Studio, yfinance, and advanced NLP techniques.

##  Features

- **Real-time Stock Data**: Live price feeds, historical charts, and financial metrics
- **Local AI Analysis**: LLM-generated insights using your local Llama 2.7B model via LM Studio
- **Multi-Agent Architecture**: Specialized agents for data analysis, sentiment, and risk assessment
- **Interactive Charts**: Candlestick charts with volume indicators using Plotly
- **Sentiment Analysis**: News sentiment scoring using TextBlob and LLM analysis
- **Risk Assessment**: 30-day volatility calculations with color-coded risk levels
- **Auto-Refresh**: Configurable live data updates
- **Connection Monitoring**: Real-time LM Studio connection status

## 🏗 Architecture

### **Your Local Setup**
- **Model Path**: TheBloke\Llama-2-7B-Chat-GGUF`
- **LM Studio API**: OpenAI-compatible endpoint on `http://localhost:1`
- **No Docker Required**: Uses your existing LM Studio installation

### Front-End Components
- **Streamlit Application** (`main-lmstudio.py`): Enhanced dashboard with LM Studio integration
- **Connection Status**: Real-time monitoring of LM Studio API availability
- **Plotly Charts**: Interactive candlestick price charts and volume indicators

### Back-End Components
- **Data Fetching Module** (`data_fetcher.py`): Integrates yfinance and News API
- **LM Studio Agent System** (`agents-lmstudio.py`): Multi-agent pattern optimized for LM Studio:
  - `DataAgent`: Stock analysis using your local Llama 2.7B model
  - `SentimentAgent`: Combined TextBlob + LLM sentiment analysis
  - `RiskAgent`: Volatility calculations with AI risk interpretation
- **OpenAI-Compatible API**: Uses standard chat completions format

## 📋 Prerequisites

- **Python 3.13+** (tested with 3.13, works with 3.9+)
- **LM Studio** installed with your model already downloaded
- **Your Model**: `TheBloke/Llama-2-7B-Chat-GGUF` (already in your system)
- **4GB+ RAM** (for running the model if not already running)

## 🔧 Installation & Setup

### Step 1: Prepare Project Files

```bash
# Create project directory
mkdir financial-ai-dashboard
cd financial-ai-dashboard


### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv financial_ai_env

# Activate virtual environment
# On Windows:
financial_ai_env\Scripts\activate
# On macOS/Linux:
source financial_ai_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download TextBlob corpora
python -m textblob.download_corpora
```

### Step 3: Configure Environment Variables

```bash
# Copy the LM Studio environment template
copy env-lmstudio.txt .env

# The .env file should contain:
LM_STUDIO_ENDPOINT=http://localhost:1
LLM_MODEL=TheBloke/Llama-2-7B-Chat-GGUF
NEWS_API_KEY=your_news_api_key_here
```

### Step 4: Start LM Studio

** Critical Step - Start Your LM Studio Server:**

1. **Open LM Studio**
2. **Load Your Model**: 
   - Navigate to your model: `TheBloke/Llama-2-7B-Chat-GGUF`
   - Located at: `TheBloke\Llama-2-7B-Chat-GGUF`
3. **Enable API Server**:
   - Go to **Server** tab in LM Studio
   - Click **Start Server**
   - Ensure it's running on **port 1**
   - You should see "Server running on http://localhost:1"

4. **Verify API Connection**:
```bash
# Test the API endpoint
curl http://localhost:1/v1/models
# Should return JSON with your model information
```

### Step 5: Run the Financial Dashboard

```bash
# Start the dashboard (make sure LM Studio server is running first!)
streamlit run main.py

# The app will open at http://localhost:85
```

## 🎯 Usage Guide

### **LM Studio Integration Features**

1. **Connection Status**: 
   - Green indicator: LM Studio connected 
   - Red indicator: LM Studio disconnected 
   - Real-time monitoring in sidebar

2. **AI Analysis Tabs**:
   - **Overview**: AI-powered stock analysis using your local Llama model
   - **Sentiment**: Enhanced news sentiment with LLM interpretation
   - **Risk**: AI risk analysis with volatility explanations

3. **Fallback Mode**:
   - If LM Studio disconnects, app continues with statistical analysis
   - Clear indicators show when AI features are unavailable

### **Dashboard Workflow**

1. **Ensure LM Studio is Running** (check green status indicator)
2. **Enter Stock Symbol** (e.g., "AAPL", "TSLA", "MSFT")
3. **Overview Tab**: View AI analysis powered by your local model
4. **Charts Tab**: Interactive price and volume analysis
5. **Sentiment Tab**: News sentiment + AI interpretation
6. **Risk Tab**: Volatility metrics with AI risk assessment

## 🔧 LM Studio Configuration

### **Optimal Settings for Financial Analysis**

```bash
# In LM Studio Server Settings:
Context Length: 4096
GPU Layers: 20-30 (adjust based on your GPU)
Batch Size: 512
Temperature: 0.7 (for balanced creativity/accuracy)
```

### **Model Performance Tips**

- **GPU Acceleration**: Enable GPU layers in LM Studio for faster responses
- **Context Window**: 4096 tokens is sufficient for financial analysis
- **Memory Usage**: Monitor RAM usage if running other applications
- **Response Time**: Expect 2-5 seconds per AI analysis depending on hardware


### **Performance Optimization**

- **Hardware**: 16GB+ RAM recommended for smooth operation
- **GPU**: NVIDIA GPU with 6GB+ VRAM for optimal performance
- **Settings**: Adjust LM Studio GPU layers based on available VRAM
- **Monitoring**: Watch LM Studio's performance metrics

##  Technical Details

### **LM Studio Integration**
- **API Format**: OpenAI-compatible chat completions
- **Connection**: HTTP requests to `localhost:1`
- **Model**: Your local `Llama-2-7B-Chat-GGUF`
- **Fallback**: Graceful degradation when LM Studio unavailable

### **Data Flow**
1. User inputs stock symbol
2. `SimpleDataFetcher` retrieves financial data
3. **LM Studio agents** process data using your local model
4. Results displayed in interactive dashboard
5. Real-time connection monitoring ensures reliability

### **Dependencies**
- **No Docker Required**: Direct integration with your LM Studio setup
- **Core**: Streamlit, Pandas, Plotly, yfinance
- **AI**: TextBlob (always available) + LM Studio (when connected)
- **Networking**: Direct HTTP calls to LM Studio API

##  Advanced Usage

### **Custom Prompts**

You can modify the AI prompts in `agents-lmstudio.py`:

```python
# Example: Customize stock analysis prompt
prompt = f"""Analyze {symbol} stock with focus on:
- Technical indicators
- Market sentiment
- Risk factors
... (customize as needed)"""
```

### **Model Switching**

If you have other models in LM Studio:

```bash
# Update .env file:
LLM_MODEL=your-other-model-name
```

### **Performance Monitoring**

The dashboard includes:
- **Connection Status**: Real-time LM Studio monitoring
- **Response Times**: Visible loading indicators
- **Error Handling**: Graceful fallback mechanisms


When running successfully, you'll see:
-  **Green Status**: "LM Studio: Connected" in sidebar
- **AI Analysis Sections**: Enhanced with "LM Studio Analysis" labels  
-  **Fast Responses**: 2-5 second AI analysis responses
-  **Rich Insights**: Detailed stock, sentiment, and risk analysis

##  Next Steps

1. **Start LM Studio** with your model loaded
2. **Run the dashboard**: `streamlit run main.py`
3. **Test with sample stock**: Try "AAPL" or "TSLA"
4. **Verify AI features**: Check for green connection status
5. **Enjoy AI-powered financial analysis!**

## Support

**LM Studio Issues**:
- Ensure server is running on port 1234
- Check model is loaded properly
- Verify sufficient RAM/GPU memory

**Dashboard Issues**:
- Check Python environment and dependencies
- Verify `.env` configuration
- Look for error messages in terminal

---
