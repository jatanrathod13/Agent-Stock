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
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import traceback

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

def calculate_rsi(data: pd.Series, window: int = 14) -> float:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else 50.0

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
            sma50 = round(history['Close'].iloc[-50:].mean(), 2)
            sma200 = round(history['Close'].iloc[-200:].mean(), 2)
            rsi = calculate_rsi(history['Close'])
            support = round(history['Close'].iloc[-200:].min(), 2)
            resistance = round(history['Close'].iloc[-200:].max(), 2)
            send_message("TechnicalAnalyzer", {
                "sma50": sma50,
                "sma200": sma200,
                "rsi": rsi,
                "support": support,
                "resistance": resistance
            })
        except Exception as e:
            logger.error(f"TechnicalAnalyzer error for {stock_symbol}: {traceback.format_exc()}")
            send_message("TechnicalAnalyzer", {"error": str(e)})

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

    def run(self, stock_symbol: str) -> None:
        try:
            stock = yf.Ticker(stock_symbol)
            news = stock.news
            if news:
                sentiments = []
                for item in news:
                    title = item.get('title', '')
                    inputs = self.tokenizer(title, return_tensors="pt", truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    sentiment = torch.argmax(probs, dim=-1).item()
                    sentiments.append(sentiment - 1)
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
                sentiment_label = "Positive" if avg_sentiment > 0.5 else "Negative" if avg_sentiment < -0.5 else "Neutral"
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
            stock_returns = stock_data.pct_change().dropna()
            volatility = round(float(stock_returns.std() * np.sqrt(252) * 100), 2)
            beta = calculate_beta(stock_symbol)
            var_95 = round(float(np.percentile(stock_returns, 5) * 100), 2)
            send_message("RiskAnalyzer", {
                "annualized_volatility": volatility,
                "beta": beta,
                "var_95": var_95
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
- Support: ${agent_data.get('TechnicalAnalyzer', {}).get('support', 'N/A')}
- Resistance: ${agent_data.get('TechnicalAnalyzer', {}).get('resistance', 'N/A')}

**Sentiment:**
- Sentiment: {agent_data.get('SentimentAnalyzer', {}).get('sentiment', 'N/A')} (source: recent news)
- Score: {agent_data.get('SentimentAnalyzer', {}).get('score', 'N/A')}

**Risk Assessment:**
- Annualized Volatility: {agent_data.get('RiskAnalyzer', {}).get('annualized_volatility', 'N/A')}%
- Beta: {agent_data.get('RiskAnalyzer', {}).get('beta', 'N/A')}
- VaR (95%): {agent_data.get('RiskAnalyzer', {}).get('var_95', 'N/A')}%

Generate a report with:
1. **Executive Summary**: Summarize findings in 2-3 sentences with a 3-month outlook.
2. **Valuation**: Compare P/E and PEG to sector peers, assess over/undervaluation.
3. **Technical Outlook**: Analyze SMA trends, RSI, and key support/resistance levels.
4. **Sentiment**: Interpret sentiment and recent news impact on short-term price.
5. **Risk Profile**: Discuss annualized volatility, beta, VaR; classify risk level.
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
                future.result()  # Wait for completion
            except Exception as e:
                logger.error(f"Agent {agent_name} failed for {stock_symbol}: {traceback.format_exc()}")
                send_message(agent_name, {"error": str(e)})

    # Collect all messages after agents complete
    agent_outputs.update(collect_all_messages())
    logger.info(f"Agent outputs after collection: {agent_outputs.keys()}")

    # Run AIResearcher with collected data
    agents["AIResearcher"].run(stock_symbol, agent_outputs)
    researcher_msg = receive_message()
    if researcher_msg:
        agent_outputs["AIResearcher"] = researcher_msg["message"]
    
    # Run TraderAgent with AIResearcher output
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