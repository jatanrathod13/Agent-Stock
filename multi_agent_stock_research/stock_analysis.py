import os
import json
from queue import Queue
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import openai
from sentence_transformers import SentenceTransformer
import faiss
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import traceback
import pandas_datareader.data as web
from stockstats import StockDataFrame
from transformers import pipeline
import torch
import torch.nn as nn
from arch import arch_model

# Suppress TOKENIZERS_PARALLELISM warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")
client = openai.OpenAI(api_key=api_key)

# Set up embedder and FAISS index
logger.info("Initializing SentenceTransformer and FAISS index")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
DIMENSION = 384
index = faiss.IndexFlatL2(DIMENSION)

# Initialize FinBERT sentiment pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Define LSTM model in PyTorch
class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Context memory functions
def store_context(message: str) -> None:
    vector = embedder.encode([message])
    index.add(np.array(vector, dtype=np.float32))

def retrieve_context(query: str, k: int = 3) -> np.ndarray:
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), k)
    return indices

def save_context_to_file(data: dict, filename: str = "context.json") -> None:
    with open(filename, "w") as file:
        json.dump(data, file)

def load_context_from_file(filename: str = "context.json") -> dict:
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Message passing system
message_queue = Queue()

def send_message(agent_name: str, message: dict) -> None:
    if isinstance(message, (pd.Series, pd.DataFrame)):
        message = message.to_dict()
    elif isinstance(message, dict):
        message = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in message.items()}
    logger.info(f"Sending message from {agent_name}: {message}")
    message_queue.put({"agent": agent_name, "message": message})
    try:
        store_context(f"{agent_name}: {json.dumps(message)}")
    except TypeError as e:
        logger.error(f"Serialization error for {agent_name}: {e}")

def receive_message() -> dict | None:
    msg = message_queue.get() if not message_queue.empty() else None
    if msg:
        logger.info(f"Received message: {msg}")
    return msg

def collect_all_messages() -> dict:
    messages = {}
    while not message_queue.empty():
        msg = receive_message()
        if msg:
            messages[msg["agent"]] = msg["message"]
    logger.info(f"Collected messages: {messages.keys()}")
    return messages

# Utility functions
def calculate_beta(stock_symbol: str) -> float:
    try:
        end = datetime.now()
        start = end - timedelta(days=365)
        stock_data = yf.download(stock_symbol, start=start, end=end, auto_adjust=False)['Adj Close']
        market_data = yf.download('^GSPC', start=start, end=end, auto_adjust=False)['Adj Close']
        common_dates = stock_data.index.intersection(market_data.index)
        stock_data = stock_data[common_dates]
        market_data = market_data[common_dates]
        if len(stock_data) < 20 or len(market_data) < 20:
            return 1.0
        stock_returns = stock_data.pct_change().dropna()
        market_returns = market_data.pct_change().dropna()
        if len(stock_returns) < 20 or len(market_returns) < 20:
            return 1.0
        cov = stock_returns.cov(market_returns)
        var = market_returns.var()
        if var == 0:
            return 1.0
        return round(cov / var, 2)
    except Exception as e:
        logger.error(f"Error calculating beta for {stock_symbol}: {str(e)}")
        return 1.0

SECTOR_PE = {
    "Technology": 25,
    "Healthcare": 20,
    "Financial Services": 15,
    "Consumer Cyclical": 18,
    "Industrials": 16,
    "Energy": 12,
    "Utilities": 14,
    "Real Estate": 20,
    "Basic Materials": 15,
    "Consumer Defensive": 18,
    "Communication Services": 20
}

def get_sector_pe(sector: str) -> float:
    return SECTOR_PE.get(sector, 15)

def generate_response(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst providing detailed stock analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {traceback.format_exc()}")
        return f"Error generating response: {str(e)}"

# Define Agents
class DataCollector:
    def run(self, stock_symbol: str) -> None:
        try:
            stock = yf.Ticker(stock_symbol)
            history = stock.history(period="1d")
            if history.empty:
                raise ValueError("No data available for the given symbol")
            price = round(history['Close'].iloc[-1], 2)
            high = round(history['High'].iloc[-1], 2)
            low = round(history['Low'].iloc[-1], 2)
            volume = int(history['Volume'].iloc[-1])
            send_message("DataCollector", {
                "current_price": price,
                "high": high,
                "low": low,
                "volume": volume
            })
        except Exception as e:
            logger.error(f"DataCollector error for {stock_symbol}: {traceback.format_exc()}")
            send_message("DataCollector", {"error": str(e)})

class FundamentalAnalyzer:
    def run(self, stock_symbol: str) -> None:
        try:
            stock = yf.Ticker(stock_symbol)
            info = stock.info
            pe_ratio = round(info.get("trailingPE", 15), 2)
            peg_ratio = info.get("pegRatio", "N/A")
            eps = round(info.get("trailingEps", 0), 2)
            market_cap = info.get("marketCap", "N/A")
            dividend_yield = round(info.get("dividendYield", 0) * 100, 2)
            debt_to_equity = info.get("debtToEquity", "N/A")
            sector = info.get("sector", "N/A")
            send_message("FundamentalAnalyzer", {
                "pe_ratio": pe_ratio,
                "peg_ratio": peg_ratio,
                "eps": eps,
                "market_cap": market_cap,
                "dividend_yield": dividend_yield,
                "debt_to_equity": debt_to_equity,
                "sector": sector
            })
        except Exception as e:
            logger.error(f"FundamentalAnalyzer error for {stock_symbol}: {traceback.format_exc()}")
            send_message("FundamentalAnalyzer", {"error": str(e)})

class TechnicalAnalyzer:
    def run(self, stock_symbol: str) -> None:
        try:
            stock = yf.Ticker(stock_symbol)
            history = stock.history(period="200d")
            if len(history) < 50:
                raise ValueError("Insufficient historical data for technical analysis")
            
            # Static indicators with stockstats
            stock_df = StockDataFrame.retype(history)
            sma50 = round(stock_df['close_50_sma'].iloc[-1], 2)
            sma200 = round(stock_df['close_200_sma'].iloc[-1], 2)
            rsi = round(stock_df['rsi_14'].iloc[-1], 2)
            macd = round(stock_df['macd'].iloc[-1], 2)
            boll_mid = round(stock_df['boll'].iloc[-1], 2)
            boll_upper = round(stock_df['boll_ub'].iloc[-1], 2)
            boll_lower = round(stock_df['boll_lb'].iloc[-1], 2)
            support = round(history['Close'].iloc[-200:].min(), 2)
            resistance = round(history['Close'].iloc[-200:].max(), 2)
            
            # LSTM price prediction with PyTorch
            prices = history['Close'].values
            scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))  # Simple min-max scaler
            scaled_prices = scaler(prices)
            sequence_length = 60
            X, y = [], []
            for i in range(sequence_length, len(scaled_prices)):
                X.append(scaled_prices[i-sequence_length:i])
                y.append(scaled_prices[i])
            X, y = np.array(X), np.array(y)
            if len(X) > 0:
                X = torch.FloatTensor(X).unsqueeze(-1)  # Shape: [samples, sequence_length, 1]
                y = torch.FloatTensor(y).unsqueeze(-1)  # Shape: [samples, 1]
                model = LSTMPricePredictor()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                for _ in range(5):  # Minimal epochs for demo; increase for better results
                    model.train()
                    optimizer.zero_grad()
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    last_sequence = torch.FloatTensor(scaled_prices[-sequence_length:]).unsqueeze(0).unsqueeze(-1)
                    predicted_price_scaled = model(last_sequence).item()
                    predicted_price = predicted_price_scaled * (np.max(prices) - np.min(prices)) + np.min(prices)
            else:
                predicted_price = "N/A"
            
            send_message("TechnicalAnalyzer", {
                "sma50": sma50,
                "sma200": sma200,
                "rsi": rsi,
                "macd": macd,
                "boll_mid": boll_mid,
                "boll_upper": boll_upper,
                "boll_lower": boll_lower,
                "support": support,
                "resistance": resistance,
                "predicted_price": round(predicted_price, 2) if isinstance(predicted_price, float) else "N/A"
            })
        except Exception as e:
            logger.error(f"TechnicalAnalyzer error for {stock_symbol}: {traceback.format_exc()}")
            send_message("TechnicalAnalyzer", {"error": str(e)})

class SentimentAnalyzer:
    def run(self, stock_symbol: str) -> None:
        try:
            stock = yf.Ticker(stock_symbol)
            news = stock.news
            if news:
                sentiments = []
                for item in news:
                    title = item.get('title', '')
                    result = sentiment_analyzer(title)[0]
                    score = 1 if result['label'] == 'Positive' else -1 if result['label'] == 'Negative' else 0
                    sentiments.append(score)
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
                sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                send_message("SentimentAnalyzer", {"sentiment": sentiment_label, "score": round(avg_sentiment, 2)})
            else:
                send_message("SentimentAnalyzer", {"sentiment": "No news available", "score": 0.0})
        except Exception as e:
            logger.error(f"SentimentAnalyzer error for {stock_symbol}: {traceback.format_exc()}")
            send_message("SentimentAnalyzer", {"error": str(e)})

class RiskAnalyzer:
    def run(self, stock_symbol: str) -> None:
        try:
            end = datetime.now()
            start = end - timedelta(days=365)
            stock_data = yf.download(stock_symbol, start=start, end=end, auto_adjust=False)['Adj Close']
            if len(stock_data) < 20:
                raise ValueError("Insufficient data for volatility calculation")
            stock_returns = stock_data.pct_change().dropna() * 100
            volatility = round(float(stock_returns.std() * np.sqrt(252)), 2)
            beta = calculate_beta(stock_symbol)
            var_95 = round(float(np.percentile(stock_returns, 5)), 2)
            
            model = arch_model(stock_returns, vol='Garch', p=1, q=1)
            garch_fit = model.fit(disp='off')
            garch_volatility = round(float(garch_fit.conditional_volatility[-1]), 2)
            
            treasury_yield = web.DataReader('DGS10', 'fred', start, end).iloc[-1]['DGS10'] if not web.DataReader('DGS10', 'fred', start, end).empty else "N/A"
            
            send_message("RiskAnalyzer", {
                "annualized_volatility": volatility,
                "garch_volatility": garch_volatility,
                "beta": beta,
                "var_95": var_95,
                "treasury_yield": treasury_yield
            })
        except Exception as e:
            logger.error(f"RiskAnalyzer error for {stock_symbol}: {traceback.format_exc()}")
            send_message("RiskAnalyzer", {"error": str(e)})

class AIResearcher:
    def run(self, stock_symbol: str, agent_data: dict) -> None:
        if agent_data:
            sector = agent_data.get('FundamentalAnalyzer', {}).get('sector', 'N/A')
            sector_pe = get_sector_pe(sector)
            prompt = f"""
You are a senior financial analyst tasked with producing a detailed, actionable stock analysis report for {stock_symbol}. Use the following data, addressing missing values with assumptions or caveats:

**Market Data:**
- Current Price: ${agent_data.get('DataCollector', {}).get('current_price', 'N/A')}
- High: ${agent_data.get('DataCollector', {}).get('high', 'N/A')}
- Low: ${agent_data.get('DataCollector', {}).get('low', 'N/A')}
- Volume: {agent_data.get('DataCollector', {}).get('volume', 'N/A')}

**Fundamentals:**
- P/E Ratio: {agent_data.get('FundamentalAnalyzer', {}).get('pe_ratio', 'N/A')} (compare to sector avg {sector_pe})
- PEG Ratio: {agent_data.get('FundamentalAnalyzer', {}).get('peg_ratio', 'N/A')}
- EPS: ${agent_data.get('FundamentalAnalyzer', {}).get('eps', 'N/A')}
- Market Cap: ${agent_data.get('FundamentalAnalyzer', {}).get('market_cap', 'N/A')}
- Dividend Yield: {agent_data.get('FundamentalAnalyzer', {}).get('dividend_yield', 'N/A')}%
- Debt-to-Equity: {agent_data.get('FundamentalAnalyzer', {}).get('debt_to_equity', 'N/A')}

**Technical Indicators:**
- 50-day SMA: ${agent_data.get('TechnicalAnalyzer', {}).get('sma50', 'N/A')}
- 200-day SMA: ${agent_data.get('TechnicalAnalyzer', {}).get('sma200', 'N/A')}
- RSI: {agent_data.get('TechnicalAnalyzer', {}).get('rsi', 'N/A')}
- MACD: {agent_data.get('TechnicalAnalyzer', {}).get('macd', 'N/A')}
- Bollinger Middle Band: ${agent_data.get('TechnicalAnalyzer', {}).get('boll_mid', 'N/A')}
- Bollinger Upper Band: ${agent_data.get('TechnicalAnalyzer', {}).get('boll_upper', 'N/A')}
- Bollinger Lower Band: ${agent_data.get('TechnicalAnalyzer', {}).get('boll_lower', 'N/A')}
- Support: ${agent_data.get('TechnicalAnalyzer', {}).get('support', 'N/A')}
- Resistance: ${agent_data.get('TechnicalAnalyzer', {}).get('resistance', 'N/A')}
- Predicted Price (Next Day): ${agent_data.get('TechnicalAnalyzer', {}).get('predicted_price', 'N/A')}

**Sentiment:**
- Sentiment: {agent_data.get('SentimentAnalyzer', {}).get('sentiment', 'N/A')} (source: recent news)
- Score: {agent_data.get('SentimentAnalyzer', {}).get('score', 'N/A')}

**Risk Assessment:**
- Annualized Volatility: {agent_data.get('RiskAnalyzer', {}).get('annualized_volatility', 'N/A')}%
- GARCH Volatility: {agent_data.get('RiskAnalyzer', {}).get('garch_volatility', 'N/A')}%
- Beta: {agent_data.get('RiskAnalyzer', {}).get('beta', 'N/A')}
- VaR (95%): {agent_data.get('RiskAnalyzer', {}).get('var_95', 'N/A')}%
- 10-Year Treasury Yield: {agent_data.get('RiskAnalyzer', {}).get('treasury_yield', 'N/A')}%

Generate a report with:
1. **Executive Summary**: Summarize findings in 2-3 sentences with a 3-month outlook.
2. **Valuation**: Compare P/E and PEG to sector peers, assess over/undervaluation.
3. **Technical Outlook**: Analyze SMA trends, RSI, MACD, Bollinger Bands, predicted price, and key support/resistance levels.
4. **Sentiment**: Interpret sentiment and recent news impact on short-term price.
5. **Risk Profile**: Discuss annualized volatility, GARCH volatility, beta, VaR, and economic context (e.g., treasury yield); classify risk level.
6. **Investment Thesis**: Provide BUY/SELL/HOLD with reasoning and catalysts (e.g., earnings, product launches).

Ensure the report is data-driven, concise, and avoids speculation. Use sector-specific benchmarks and estimate missing data where possible.
"""
            summary = generate_response(prompt)
            send_message("AIResearcher", {"summary": summary})
        else:
            logger.warning(f"No data available for {stock_symbol} to summarize in AIResearcher")
            send_message("AIResearcher", {"summary": "No data available to summarize."})

class TraderAgent:
    def run(self, stock_symbol: str, researcher_data: dict) -> None:
        summary = researcher_data.get("summary", "No data available to summarize.")
        prompt = f"""
You are an experienced trader evaluating this stock analysis report for {stock_symbol}:

{summary}

Provide an investment recommendation in this format:
- Recommendation: [BUY/SELL/HOLD] (3-6 months)
- Justification: [2-3 sentences weighing valuation, technicals, sentiment, risk, and catalysts]

Base your decision on reliable metrics (e.g., P/E, SMA trends, beta) and assume a 3-6 month horizon. Specify assumptions for missing data and highlight key catalysts (e.g., earnings, product updates).
"""
        recommendation = generate_response(prompt)
        recommendation = recommendation.replace("\n\n", " ").replace("\n", " ").lstrip("- ")
        send_message("TraderAgent", {"recommendation": recommendation})

def analyze(stock_symbol: str) -> dict:
    global message_queue
    message_queue = Queue()
    logger.info(f"Starting analysis for {stock_symbol}")
    
    agents = {
        "DataCollector": DataCollector(),
        "FundamentalAnalyzer": FundamentalAnalyzer(),
        "TechnicalAnalyzer": TechnicalAnalyzer(),
        "SentimentAnalyzer": SentimentAnalyzer(),
        "RiskAnalyzer": RiskAnalyzer(),
        "AIResearcher": AIResearcher(),
        "TraderAgent": TraderAgent()
    }
    
    agent_outputs = {}
    with ThreadPoolExecutor() as executor:
        futures = {agent_name: executor.submit(agent.run, stock_symbol) for agent_name, agent in agents.items() if agent_name not in ["AIResearcher", "TraderAgent"]}
        for agent_name, future in futures.items():
            try:
                future.result()
            except Exception as e:
                logger.error(f"Agent {agent_name} failed for {stock_symbol}: {traceback.format_exc()}")
                send_message(agent_name, {"error": str(e)})

    agent_outputs.update(collect_all_messages())
    logger.info(f"Agent outputs after collection: {agent_outputs.keys()}")

    agents["AIResearcher"].run(stock_symbol, agent_outputs)
    researcher_msg = receive_message()
    if researcher_msg:
        agent_outputs["AIResearcher"] = researcher_msg["message"]
    
    if "AIResearcher" in agent_outputs:
        agents["TraderAgent"].run(stock_symbol, agent_outputs["AIResearcher"])
        trader_msg = receive_message()
        if trader_msg:
            agent_outputs["TraderAgent"] = trader_msg["message"]
    
    final_report = agent_outputs.get("TraderAgent", {}).get("recommendation", "Analysis failed.")
    return {
        "stock": stock_symbol,
        "report": final_report,
        "agent_outputs": agent_outputs
    }

if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn
    app = FastAPI(title="Stock Analysis API", description="Multi-agent stock analysis system")
    @app.post("/analyze/{stock_symbol}")
    def api_analyze(stock_symbol: str) -> dict:
        return analyze(stock_symbol)
    uvicorn.run(app, host="0.0.0.0", port=8090)