import requests
import json
import os
from typing import List, Dict, Tuple, Optional
from textblob import TextBlob
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class SimpleAgent(ABC):
    """
    Base class for all AI agents in the financial intelligence system
    Compatible with LM Studio local API server
    """
    
    def __init__(self):
        """Initialize base agent for LM Studio"""
        self.lm_studio_endpoint = os.getenv('LM_STUDIO_ENDPOINT', 'http://localhost:1234')
        self.model_name = os.getenv('LLM_MODEL', 'TheBloke/Llama-2-7B-Chat-GGUF')
        self.timeout = 30
    
    def _call_lm_studio(self, prompt: str) -> str:
        """
        Make HTTP call to LM Studio API for LLM inference
        Uses OpenAI-compatible API format
        
        Args:
            prompt (str): Input prompt for the LLM
            
        Returns:
            str: LLM response
        """
        try:
            url = f"{self.lm_studio_endpoint}/v1/chat/completions"
            
            # Format prompt for chat completion
            messages = [
                {
                    "role": "system", 
                    "content": "You are a professional financial analyst. Provide concise, accurate analysis."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 200,
                "stream": False
            }
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.timeout,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content'].strip()
                else:
                    return self._fallback_response()
            else:
                print(f"LM Studio API error: {response.status_code} - {response.text}")
                return self._fallback_response()
                
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error to LM Studio: {str(e)}")
            print("Make sure LM Studio is running with the API server enabled on port 1234")
            return self._fallback_response()
        except requests.exceptions.RequestException as e:
            print(f"Request error to LM Studio: {str(e)}")
            return self._fallback_response()
        except Exception as e:
            print(f"Error calling LM Studio API: {str(e)}")
            return self._fallback_response()
    
    @abstractmethod
    def _fallback_response(self) -> str:
        """
        Provide fallback response when LLM is unavailable
        
        Returns:
            str: Fallback response
        """
        pass

class DataAgent(SimpleAgent):
    """
    Agent responsible for analyzing stock data and providing insights
    """
    
    def analyze_stock(self, symbol: str, current_price: float, market_cap: float, pe_ratio: float) -> str:
        """
        Analyze stock data and provide AI-generated insights
        
        Args:
            symbol (str): Stock ticker symbol
            current_price (float): Current stock price
            market_cap (float): Market capitalization
            pe_ratio (float): Price-to-earnings ratio
            
        Returns:
            str: AI analysis of the stock
        """
        # Create analysis prompt
        market_cap_display = f"${market_cap/1e9:.2f}B" if market_cap else "N/A"
        pe_ratio_display = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
        
        prompt = f"""Analyze {symbol} stock with these metrics:
- Current Price: ${current_price:.2f}
- Market Cap: {market_cap_display}
- P/E Ratio: {pe_ratio_display}

Provide exactly 2 sentences covering:
1. Valuation assessment and market position
2. Investment outlook or key consideration

Be professional and concise."""
        
        return self._call_lm_studio(prompt)
    
    def analyze_price_trend(self, price_data: pd.DataFrame, symbol: str) -> str:
        """
        Analyze price trend over time
        
        Args:
            price_data (pd.DataFrame): Historical price data
            symbol (str): Stock ticker symbol
            
        Returns:
            str: Trend analysis
        """
        if price_data is None or len(price_data) < 2:
            return self._fallback_response()
        
        # Calculate basic trend metrics
        current_price = price_data['Close'].iloc[-1]
        start_price = price_data['Close'].iloc[0]
        price_change_pct = ((current_price - start_price) / start_price) * 100
        
        # Calculate recent volatility
        returns = price_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        prompt = f"""Analyze {symbol} price trend:
- Price change: {price_change_pct:.2f}%
- Current price: ${current_price:.2f}
- Volatility: {volatility:.2f}

Provide 2 sentences on trend direction and volatility assessment."""
        
        return self._call_lm_studio(prompt)
    
    def _fallback_response(self) -> str:
        """Fallback response for data analysis"""
        return "The stock demonstrates typical market behavior with regular price fluctuations. Monitor key support and resistance levels for potential trading opportunities based on current market conditions."

class SentimentAgent(SimpleAgent):
    """
    Agent responsible for sentiment analysis of news and social media
    """
    
    def analyze_sentiment(self, news_data: List[Dict]) -> float:
        """
        Analyze sentiment from news articles using TextBlob
        
        Args:
            news_data (List[Dict]): List of news articles
            
        Returns:
            float: Sentiment score between -1 and 1
        """
        if not news_data:
            return 0.0
        
        sentiments = []
        
        for article in news_data:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            if text.strip():
                try:
                    blob = TextBlob(text)
                    sentiments.append(blob.sentiment.polarity)
                except Exception as e:
                    print(f"Error analyzing sentiment: {str(e)}")
                    continue
        
        # Return average sentiment
        return sum(sentiments) / len(sentiments) if sentiments else 0.0
    
    def analyze_news_sentiment_llm(self, news_data: List[Dict], symbol: str) -> str:
        """
        Use LLM to analyze sentiment from news data
        
        Args:
            news_data (List[Dict]): List of news articles
            symbol (str): Stock ticker symbol
            
        Returns:
            str: LLM-generated sentiment analysis
        """
        if not news_data:
            return "No recent news available for sentiment analysis."
        
        # Prepare news headlines for analysis
        headlines = []
        for article in news_data[:5]:  # Limit to top 5 articles
            title = article.get('title', '')
            if title:
                headlines.append(f"• {title}")
        
        if not headlines:
            return "No valid headlines found for analysis."
        
        news_text = "\n".join(headlines)
        
        prompt = f"""Analyze news sentiment for {symbol}:

{news_text}

Provide 2 sentences:
1. Overall sentiment classification (positive/negative/neutral)
2. Key factors driving the sentiment"""
        
        return self._call_lm_studio(prompt)
    
    def get_sentiment_summary(self, sentiment_score: float) -> Dict[str, str]:
        """
        Get sentiment summary based on score
        
        Args:
            sentiment_score (float): Sentiment score between -1 and 1
            
        Returns:
            Dict with sentiment classification and description
        """
        if sentiment_score > 0.1:
            return {
                "classification": "Positive",
                "description": "Recent news shows generally positive sentiment",
                "color": "green",
                "emoji": "📈"
            }
        elif sentiment_score < -0.1:
            return {
                "classification": "Negative", 
                "description": "Recent news shows generally negative sentiment",
                "color": "red",
                "emoji": "📉"
            }
        else:
            return {
                "classification": "Neutral",
                "description": "Recent news shows neutral or mixed sentiment", 
                "color": "yellow",
                "emoji": "📊"
            }
    
    def _fallback_response(self) -> str:
        """Fallback response for sentiment analysis"""
        return "News sentiment analysis indicates mixed market signals with no strong directional bias detected in recent coverage."

class RiskAgent(SimpleAgent):
    """
    Agent responsible for risk assessment and volatility calculations
    """
    
    def calculate_risk(self, price_data: pd.DataFrame, window: int = 30) -> Tuple[Optional[float], str]:
        """
        Calculate risk metrics based on price volatility
        
        Args:
            price_data (pd.DataFrame): Historical price data
            window (int): Rolling window for volatility calculation
            
        Returns:
            Tuple of (volatility, risk_level)
        """
        if price_data is None or len(price_data) < window:
            return None, "Unknown"
        
        try:
            # Calculate daily returns
            returns = price_data['Close'].pct_change().dropna()
            
            # Calculate 30-day rolling volatility
            volatility = returns.rolling(window=window).std().iloc[-1]
            
            # Determine risk level based on volatility
            if volatility < 0.02:  # Less than 2% daily volatility
                risk_level = "Low"
            elif volatility < 0.04:  # Less than 4% daily volatility
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return volatility, risk_level
            
        except Exception as e:
            print(f"Error calculating risk: {str(e)}")
            return None, "Unknown"
    
    def calculate_value_at_risk(self, price_data: pd.DataFrame, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) metrics
        
        Args:
            price_data (pd.DataFrame): Historical price data
            confidence_level (float): Confidence level for VaR calculation
            
        Returns:
            Dict containing VaR metrics
        """
        if price_data is None or len(price_data) < 30:
            return {}
        
        try:
            # Calculate daily returns
            returns = price_data['Close'].pct_change().dropna()
            current_price = price_data['Close'].iloc[-1]
            
            # Calculate VaR at different time horizons
            daily_var = np.percentile(returns, (1 - confidence_level) * 100)
            weekly_var = daily_var * np.sqrt(5)  # Assuming 5 trading days per week
            monthly_var = daily_var * np.sqrt(21)  # Assuming 21 trading days per month
            
            return {
                'daily_var': daily_var,
                'weekly_var': weekly_var,
                'monthly_var': monthly_var,
                'daily_var_amount': current_price * daily_var,
                'weekly_var_amount': current_price * weekly_var,
                'monthly_var_amount': current_price * monthly_var,
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            print(f"Error calculating VaR: {str(e)}")
            return {}
    
    def analyze_risk_llm(self, volatility: float, risk_level: str, symbol: str) -> str:
        """
        Use LLM to provide risk analysis
        
        Args:
            volatility (float): Calculated volatility
            risk_level (str): Risk level classification
            symbol (str): Stock ticker symbol
            
        Returns:
            str: LLM-generated risk analysis
        """
        prompt = f"""Risk analysis for {symbol}:
- 30-day volatility: {volatility:.4f}
- Risk level: {risk_level}

Provide 2 sentences explaining:
1. What this risk level means for investors
2. Practical investment implications or recommendations"""
        
        return self._call_lm_studio(prompt)
    
    def get_risk_recommendations(self, risk_level: str) -> Dict[str, str]:
        """
        Get risk-based investment recommendations
        
        Args:
            risk_level (str): Risk level classification
            
        Returns:
            Dict with recommendations
        """
        recommendations = {
            "Low": {
                "strategy": "Conservative Investment",
                "description": "Suitable for risk-averse investors seeking stable returns",
                "allocation": "Can be a core holding in diversified portfolios"
            },
            "Medium": {
                "strategy": "Balanced Investment",
                "description": "Appropriate for moderate risk tolerance with growth potential",
                "allocation": "Consider position sizing and stop-loss strategies"
            },
            "High": {
                "strategy": "Aggressive Investment",
                "description": "Only suitable for high-risk tolerance investors",
                "allocation": "Limit position size and use risk management tools"
            },
            "Unknown": {
                "strategy": "Cautious Approach",
                "description": "Insufficient data for proper risk assessment",
                "allocation": "Conduct additional research before investing"
            }
        }
        
        return recommendations.get(risk_level, recommendations["Unknown"])
    
    def _fallback_response(self) -> str:
        """Fallback response for risk analysis"""
        return "Risk assessment indicates normal market volatility patterns. Consider your personal risk tolerance and investment timeline when making portfolio decisions."