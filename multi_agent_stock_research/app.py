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
from fastapi import FastAPI
import uvicorn
import logging
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = openai.OpenAI(api_key=api_key)

# Initialize embedder and FAISS index for context memory
embedder = SentenceTransformer("all-MiniLM-L6-v2")
dim = 384
index = faiss.IndexFlatL2(dim)

# Context memory functions
def store_context(message):
    vector = embedder.encode([message])
    index.add(np.array(vector, dtype=np.float32))

def retrieve_context(query, k=3):
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), k)
    return indices

def save_context_to_file(data, filename="context.json"):
    with open(filename, "w") as file:
        json.dump(data, file)

def load_context_from_file(filename="context.json"):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Message passing with Queue
message_queue = Queue()

def send_message(agent_name, message):
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

def receive_message():
    msg = message_queue.get() if not message_queue.empty() else None
    if msg:
        logger.info(f"Received message: {msg}")
    else:
        logger.warning("Queue is empty when receiving message")
    return msg

# Helper function for beta calculation
def calculate_beta(stock_symbol):
    try:
        end = datetime.now()
        start = end - timedelta(days=365)
        stock_data = yf.download(stock_symbol, start=start, end=end, auto_adjust=False)['Adj Close']
        market_data = yf.download('^GSPC', start=start, end=end, auto_adjust=False)['Adj Close']
        common_dates = stock_data.index.intersection(market_data.index)
        stock_data = stock_data[common_dates]
        market_data = market_data[common_dates]
        if len(stock_data) < 20 or len(market_data) < 20:
            return "N/A"
        stock_returns = stock_data.pct_change().dropna()
        market_returns = market_data.pct_change().dropna()
        if len(stock_returns) < 20 or len(market_returns) < 20:
            return "N/A"
        cov = stock_returns.cov(market_returns)
        var = market_returns.var()
        if var == 0:
            return "N/A"
        beta = cov / var
        return round(beta, 2)
    except Exception as e:
        logger.error(f"Error calculating beta for {stock_symbol}: {str(e)}")
        return "N/A"

# Function to generate GPT-4 response
def generate_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst providing detailed stock analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Define Agents
class DataCollector:
    def run(self, stock_symbol):
        try:
            stock = yf.Ticker(stock_symbol)
            history = stock.history(period="1d")
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
            send_message("DataCollector", {"error": str(e)})

class FundamentalAnalyzer:
    def run(self, stock_symbol):
        try:
            stock = yf.Ticker(stock_symbol)
            info = stock.info
            pe_ratio = info.get("trailingPE", "N/A")
            if isinstance(pe_ratio, (int, float)):
                pe_ratio = round(pe_ratio, 2)
            eps = info.get("trailingEps", "N/A")
            if isinstance(eps, (int, float)):
                eps = round(eps, 2)
            market_cap = info.get("marketCap", "N/A")
            dividend_yield = info.get("dividendYield", "N/A")
            if isinstance(dividend_yield, (int, float)):
                dividend_yield = round(dividend_yield * 100, 2)  # Convert to percentage
            send_message("FundamentalAnalyzer", {
                "pe_ratio": pe_ratio,
                "eps": eps,
                "market_cap": market_cap,
                "dividend_yield": dividend_yield
            })
        except Exception as e:
            send_message("FundamentalAnalyzer", {"error": str(e)})

class TechnicalAnalyzer:
    def run(self, stock_symbol):
        try:
            stock = yf.Ticker(stock_symbol)
            history = stock.history(period="200d")
            sma50 = round(history['Close'].iloc[-50:].mean(), 2)
            sma200 = round(history['Close'].iloc[-200:].mean(), 2)
            send_message("TechnicalAnalyzer", {
                "sma50": sma50,
                "sma200": sma200
            })
        except Exception as e:
            send_message("TechnicalAnalyzer", {"error": str(e)})

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

    def run(self, stock_symbol):
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
                    sentiment = torch.argmax(probs, dim=-1).item()  # 0: Neutral, 1: Positive, 2: Negative
                    sentiments.append(sentiment - 1)  # Convert to -1 (Negative), 0 (Neutral), 1 (Positive)
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
                sentiment_label = "Positive" if avg_sentiment > 0.5 else "Negative" if avg_sentiment < -0.5 else "Neutral"
                send_message("SentimentAnalyzer", {"sentiment": sentiment_label, "score": round(avg_sentiment, 2)})
            else:
                send_message("SentimentAnalyzer", {"sentiment": "No news available", "score": 0.0})
        except Exception as e:
            send_message("SentimentAnalyzer", {"error": str(e)})

class RiskAnalyzer:
    def run(self, stock_symbol):
        try:
            end = datetime.now()
            start = end - timedelta(days=365)
            stock_data = yf.download(stock_symbol, start=start, end=end, auto_adjust=False)['Adj Close']
            if len(stock_data) < 20:
                raise ValueError("Insufficient data for volatility calculation")
            stock_returns = stock_data.pct_change().dropna()
            volatility = round(stock_returns.std(), 4)
            beta = calculate_beta(stock_symbol)
            # Placeholder for VaR; requires XGBoost model training
            var_95 = "N/A"
            send_message("RiskAnalyzer", {
                "volatility": volatility,
                "beta": beta,
                "var_95": var_95
            })
        except Exception as e:
            send_message("RiskAnalyzer", {"error": str(e)})

class AIResearcher:
    def run(self, stock_symbol):
        data = {}
        while not message_queue.empty():
            msg = receive_message()
            if msg:
                data[msg["agent"]] = msg["message"]
        
        if data:
            prompt = f"""
You are a senior financial analyst tasked with producing a detailed, actionable stock analysis report for {stock_symbol}. Use the following data, addressing missing values with assumptions or caveats:

**Market Data:**
- Current Price: ${data.get('DataCollector', {}).get('current_price', 'N/A')}
- High: ${data.get('DataCollector', {}).get('high', 'N/A')}
- Low: ${data.get('DataCollector', {}).get('low', 'N/A')}
- Volume: {data.get('DataCollector', {}).get('volume', 'N/A')}

**Fundamentals:**
- P/E Ratio: {data.get('FundamentalAnalyzer', {}).get('pe_ratio', 'N/A')} (assume industry avg 15 if N/A)
- EPS: ${data.get('FundamentalAnalyzer', {}).get('eps', 'N/A')}
- Market Cap: ${data.get('FundamentalAnalyzer', {}).get('market_cap', 'N/A')}
- Dividend Yield: {data.get('FundamentalAnalyzer', {}).get('dividend_yield', 'N/A')}%

**Technical Indicators:**
- 50-day SMA: ${data.get('TechnicalAnalyzer', {}).get('sma50', 'N/A')}
- 200-day SMA: ${data.get('TechnicalAnalyzer', {}).get('sma200', 'N/A')}

**Sentiment:**
- Sentiment: {data.get('SentimentAnalyzer', {}).get('sentiment', 'N/A')}
- Sentiment Score: {data.get('SentimentAnalyzer', {}).get('score', 'N/A')}

**Risk Assessment:**
- Volatility: {data.get('RiskAnalyzer', {}).get('volatility', 'N/A')}
- Beta: {data.get('RiskAnalyzer', {}).get('beta', 'N/A')}
- VaR (95%): {data.get('RiskAnalyzer', {}).get('var_95', 'N/A')}

Generate a report with these sections:
1. **Executive Summary**: Summarize key findings in 2-3 sentences.
2. **Valuation**: Compare P/E to industry avg (assume 15 if unknown), assess if over/undervalued, and interpret dividend yield.
3. **Technical Outlook**: Analyze SMA crossover (50-day vs 200-day).
4. **Sentiment**: Weigh sentiment score and label; suggest implications for short-term price movement.
5. **Risk Profile**: Interpret volatility, beta, and VaR; classify risk as Low/Moderate/High.
6. **Investment Thesis**: Provide a BUY, SELL, or HOLD recommendation with clear reasoning tied to data.

Ensure the report is concise, data-driven, and handles missing data explicitly (e.g., 'P/E unavailable, assuming industry avg of 15'). Avoid speculation beyond the data provided.
"""
            summary = generate_response(prompt)
            send_message("AIResearcher", {"summary": summary})
        else:
            send_message("AIResearcher", {"summary": "No data available to summarize."})

class TraderAgent:
    def run(self, stock_symbol):
        summary = None
        while not message_queue.empty():
            msg = receive_message()
            if msg and msg["agent"] == "AIResearcher":
                summary = msg["message"]["summary"]
        
        if summary:
            prompt = f"""
You are an experienced trader evaluating this stock analysis report for {stock_symbol}:

{summary}

Provide an investment recommendation in this exact format:
- Recommendation: [BUY/SELL/HOLD]
- Justification: [2-3 sentences explaining your decision, weighing valuation, technicals, sentiment, and risk]

Base your decision on the report’s data, prioritizing reliable metrics (e.g., P/E, SMA crossover, beta) over uncertain ones (e.g., missing VaR). If data is incomplete, state assumptions and focus on actionable insights for a 3-month horizon.
"""
            recommendation = generate_response(prompt)
            recommendation = recommendation.replace("\n\n", " ").replace("\n", " ").lstrip("- ")
            send_message("TraderAgent", {"recommendation": recommendation})
        else:
            send_message("TraderAgent", {"recommendation": "No data available for recommendation."})

# FastAPI app
app = FastAPI()

@app.post("/analyze/{stock_symbol}")
def analyze(stock_symbol: str):
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
    for agent_name, agent in list(agents.items())[:-2]:  # Run all except AIResearcher and TraderAgent
        logger.info(f"Running {agent_name}")
        agent.run(stock_symbol)
        msg = receive_message()
        if msg:
            agent_outputs[agent_name] = msg["message"]
    
    # Re-populate queue for AIResearcher
    for agent_name, output in agent_outputs.items():
        send_message(agent_name, output)
    
    # Run AIResearcher
    logger.info("Running AIResearcher")
    agents["AIResearcher"].run(stock_symbol)
    researcher_msg = receive_message()
    if researcher_msg:
        agent_outputs["AIResearcher"] = researcher_msg["message"]
    
    # Ensure queue is populated for TraderAgent
    if "AIResearcher" in agent_outputs:
        send_message("AIResearcher", agent_outputs["AIResearcher"])
    
    # Run TraderAgent
    logger.info("Running TraderAgent")
    agents["TraderAgent"].run(stock_symbol)
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
    uvicorn.run(app, host="0.0.0.0", port=8090)