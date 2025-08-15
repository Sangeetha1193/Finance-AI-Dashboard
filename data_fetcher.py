import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional

class SimpleDataFetcher:
    """
    Simple data fetcher class for retrieving stock data and news
    """

    def __init__(self):
        """Initialize the data fetcher"""
        self.news_api_key = os.getenv('NEWS_API_KEY', None)
        self.base_news_url = "https://newsapi.org/v2/everything"
        self.fmp_api_key = os.getenv('FMP_API_KEY', None)
        self.fmp_base_url = "https://financialmodelingprep.com/api/v3"

    def get_stock_data(self, symbol: str, period: str = "6mo") -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[str], Optional[float], Optional[float]]:
        """
        Fetch stock data using yfinance

        Args:
            symbol (str): Stock ticker symbol
            period (str): Time period for data (default: 6mo)

        Returns:
            Tuple containing:
            - DataFrame with OHLCV data
            - Current price
            - Company name
            - Market cap
            - P/E ratio
        """
        try:
            # Use yfinance to fetch data as fallback / supplementary
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period, interval="1d")

            if hist_data.empty:
                return None, None, None, None, None

            current_price = hist_data['Close'].iloc[-1]
            info = ticker.info
            company_name = info.get('longName', symbol)
            market_cap = info.get('marketCap', None)
            pe_ratio = info.get('trailingPE', None)

            return hist_data, current_price, company_name, market_cap, pe_ratio

        except Exception as e:
            print(f"Error fetching stock data for {symbol} (yfinance): {str(e)}")

        # As fallback, try FinancialModelingPrep API:
        if self.fmp_api_key:
            try:
                url = f"{self.fmp_base_url}/historical-price-full/{symbol}?timeseries=80&apikey={self.fmp_api_key}"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                if 'historical' in data and isinstance(data['historical'], list):
                    hist_list = data['historical']

                    # Convert list of dicts to DataFrame
                    df = pd.DataFrame(hist_list)
                    # Rename to match yfinance format
                    df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

                    current_price = df['Close'].iloc[0] if not df.empty else None
                    # For company info fallback, just use symbol and None
                    company_name = symbol
                    market_cap = None
                    pe_ratio = None

                    return df, current_price, company_name, market_cap, pe_ratio
                else:
                    print(f"No historical data found for {symbol} from FMP.")
                    return None, None, None, None, None

            except Exception as e:
                print(f"Error fetching FMP data for {symbol}: {str(e)}")
                return None, None, None, None, None

        # If no data found
        return None, None, None, None, None

    def get_simple_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get simple news data - either from API or mock data

        Args:
            symbol (str): Stock ticker symbol
            limit (int): Number of articles to return

        Returns:
            List of news articles
        """
        # If we have a news API key, try to fetch real news
        if self.news_api_key:
            try:
                return self._fetch_news_from_api(symbol, limit)
            except Exception as e:
                print(f"Error fetching news from API: {str(e)}")
                return self._get_mock_news(symbol, limit)
        else:
            # Return mock news data
            return self._get_mock_news(symbol, limit)

    def _fetch_news_from_api(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Fetch real news from NewsAPI

        Args:
            symbol (str): Stock ticker symbol
            limit (int): Number of articles to return

        Returns:
            List of news articles
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        params = {
            'q': f'{symbol} OR stock OR market',
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'pageSize': limit,
            'apiKey': self.news_api_key,
            'language': 'en'
        }

        response = requests.get(self.base_news_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        articles = []

        for article in data.get('articles', []):
            articles.append({
                'title': article.get('title', 'No title'),
                'description': article.get('description', 'No description'),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', 'Unknown')
            })

        return articles

    def _get_mock_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Generate mock news data for demonstration purposes

        Args:
            symbol (str): Stock ticker symbol
            limit (int): Number of articles to return

        Returns:
            List of mock news articles
        """
        mock_articles = [
            {
                'title': f'{symbol} Reports Strong Quarterly Earnings Beat Expectations',
                'description': f'{symbol} exceeded analyst expectations with strong revenue growth and positive outlook for the upcoming quarter.',
                'url': '#',
                'published_at': '2024-01-15T10:30:00Z',
                'source': 'Financial Times'
            },
            {
                'title': f'Market Analysis: {symbol} Shows Resilient Performance',
                'description': f'Despite market volatility, {symbol} maintains steady growth trajectory with solid fundamentals.',
                'url': '#',
                'published_at': '2024-01-14T15:45:00Z',
                'source': 'MarketWatch'
            },
            {
                'title': f'Analysts Upgrade {symbol} Rating Following Recent Developments',
                'description': f'Several major investment firms have upgraded their rating for {symbol} citing improved market conditions.',
                'url': '#',
                'published_at': '2024-01-13T09:15:00Z',
                'source': 'Bloomberg'
            },
            # Add more mock articles as needed...
        ]

        return mock_articles[:limit]
