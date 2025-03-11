import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism for tokenizers

import time
import json
from queue import Queue
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import openai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = openai.OpenAI(api_key=api_key)

def generate_response(prompt):
    """Generate a response using GPT-4 via OpenAI API (v1.0.0+ compatible)."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
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

# Shared Context Memory (FAISS + JSON)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
dim = 384
index = faiss.IndexFlatL2(dim)

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

# Message Passing with Queue
message_queue = Queue()

def send_message(agent_name, message):
    """Send a message to the queue and store context, ensuring JSON serializability."""
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

# Helper Function for Beta Calculation
def calculate_beta(stock_symbol):
    try:
        end = datetime.now()
        start = end - timedelta(days=365)
        stock_data = yf.download(stock_symbol, start=start, end=end, auto_adjust=False)['Adj Close']
        market_data = yf.download('^GSPC', start=start, end=end, auto_adjust=False)['Adj Close']
        logger.info(f"Stock data length for {stock_symbol}: {len(stock_data)}, Market data length: {len(market_data)}")
        
        common_dates = stock_data.index.intersection(market_data.index)
        stock_data = stock_data[common_dates]
        market_data = market_data[common_dates]
        if len(stock_data) < 20 or len(market_data) < 20:
            logger.warning(f"Insufficient overlapping data for {stock_symbol}: {len(stock_data)} stock, {len(market_data)} market")
            return "N/A"
        
        stock_returns = stock_data.pct_change().dropna()
        market_returns = market_data.pct_change().dropna()
        if len(stock_returns) < 20 or len(market_returns) < 20:
            logger.warning(f"Insufficient returns data for {stock_symbol}: {len(stock_returns)} stock, {len(market_returns)} market")
            return "N/A"
        
        cov = stock_returns.cov(market_returns)
        var = market_returns.var()
        if var == 0:
            logger.warning(f"Market variance is zero for {stock_symbol}")
            return "N/A"
        
        beta = cov / var
        logger.info(f"Beta calculated for {stock_symbol}: {beta}")
        return round(beta, 2)
    except Exception as e:
        logger.error(f"Error calculating beta for {stock_symbol}: {str(e)}")
        return "N/A"

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
                logger.info(f"Raw dividendYield for {stock_symbol}: {dividend_yield}")
                # Always assume yfinance returns decimal (e.g., 0.0044) and scale to percentage
                dividend_yield = round(dividend_yield * 100, 2)
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
    def run(self, stock_symbol):
        try:
            stock = yf.Ticker(stock_symbol)
            news = stock.news
            if news:
                analyzer = SentimentIntensityAnalyzer()
                scores = []
                for item in news:
                    title = item.get('title', None)
                    if title:
                        scores.append(analyzer.polarity_scores(title)['compound'])
                    else:
                        scores.append(0.0)
                avg_score = sum(scores) / len(scores) if scores else 0.0
                if avg_score > 0.05:
                    sentiment = "Positive"
                elif avg_score < -0.05:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
            else:
                sentiment = "No news available"
            send_message("SentimentAnalyzer", {"sentiment": sentiment})
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
            volatility = float(round(stock_returns.std(), 4))
            risk_level = "N/A"
            if isinstance(volatility, float):
                if volatility > 0.04:
                    risk_level = "High"
                elif volatility > 0.02:
                    risk_level = "Moderate"
                else:
                    risk_level = "Low"
            beta = calculate_beta(stock_symbol)
            send_message("RiskAnalyzer", {
                "volatility": volatility,
                "risk_level": risk_level,
                "beta": beta
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
You are a financial analyst tasked with generating a detailed stock analysis report for {stock_symbol}. Use the following data, handling missing or incomplete values by making reasonable assumptions or noting uncertainties:

**Market Data:**
- Current Price: ${data.get('DataCollector', {}).get('current_price', 'N/A')}
- High: ${data.get('DataCollector', {}).get('high', 'N/A')}
- Low: ${data.get('DataCollector', {}).get('low', 'N/A')}
- Volume: {data.get('DataCollector', {}).get('volume', 'N/A')}

**Fundamentals:**
- P/E Ratio: {data.get('FundamentalAnalyzer', {}).get('pe_ratio', 'N/A')}
- EPS: ${data.get('FundamentalAnalyzer', {}).get('eps', 'N/A')}
- Market Cap: ${data.get('FundamentalAnalyzer', {}).get('market_cap', 'N/A')}
- Dividend Yield: {data.get('FundamentalAnalyzer', {}).get('dividend_yield', 'N/A')}%

**Technical Indicators:**
- 50-day SMA: ${data.get('TechnicalAnalyzer', {}).get('sma50', 'N/A')}
- 200-day SMA: ${data.get('TechnicalAnalyzer', {}).get('sma200', 'N/A')}

**Sentiment:**
- Overall Sentiment: {data.get('SentimentAnalyzer', {}).get('sentiment', 'N/A')}

**Risk Assessment:**
- Volatility: {data.get('RiskAnalyzer', {}).get('volatility', 'N/A')}
- Risk Level: {data.get('RiskAnalyzer', {}).get('risk_level', 'N/A')}
- Beta: {data.get('RiskAnalyzer', {}).get('beta', 'N/A')}

Generate a report with the following structure:
1. **Market Data Analysis**: Interpret price, high, low, and volume trends.
2. **Fundamental Analysis**: Evaluate valuation and profitability; if data is missing or seems erroneous (e.g., unusually high dividend yield), note potential implications.
3. **Technical Analysis**: Assess short- and long-term trends using SMAs.
4. **Sentiment Analysis**: Interpret market sentiment; if unavailable, suggest monitoring news.
5. **Risk Assessment**: Analyze volatility and beta; estimate risk if data is incomplete.
6. **Summary and Recommendation**: Summarize findings and provide an investment recommendation (BUY, SELL, or HOLD) with reasoning, even if data is partial.

Ensure the report is detailed, data-driven, and actionable, acknowledging any gaps or inconsistencies in data while still providing useful insights for an investor.
"""
            summary = generate_response(prompt)
            send_message("AIResearcher", {"summary": summary})
        else:
            send_message("AIResearcher", {"summary": "No data available to summarize."})

class TraderAgent:
    def run(self, stock_symbol):
        summary = None
        logger.info(f"TraderAgent starting for {stock_symbol}, queue size: {message_queue.qsize()}")
        while not message_queue.empty():
            msg = receive_message()
            if msg and msg["agent"] == "AIResearcher":
                summary = msg["message"]["summary"]
                logger.info(f"TraderAgent received summary: {summary[:100]}...")
        
        if summary:
            prompt = f"""
You are a trader reviewing this stock analysis report for {stock_symbol}:

{summary}

Based on this analysis, provide a concise investment recommendation (BUY, SELL, or HOLD) with a brief justification in the format:
- Recommendation: [BUY/SELL/HOLD]
- Justification: [Your reasoning]

Use available data to weigh key factors (e.g., valuation, trends, sentiment, risk), and if data is incomplete or inconsistent (e.g., unusual dividend yield), base your decision on the most reliable insights while noting uncertainty. Ensure your recommendation is actionable and grounded in the report.
"""
            recommendation = generate_response(prompt)
            # Clean up newlines and remove leading dash
            recommendation = recommendation.replace("\n\n", " ").replace("\n", " ").lstrip("- ")
            send_message("TraderAgent", {"recommendation": recommendation})
        else:
            logger.warning(f"No summary available for {stock_symbol} in TraderAgent")
            send_message("TraderAgent", {"recommendation": "No data available for recommendation."})

# FastAPI Web Interface
app = FastAPI()

@app.post("/analyze/{stock_symbol}")
def analyze(stock_symbol: str):
    global message_queue
    message_queue = Queue()
    logger.info(f"Starting analysis for {stock_symbol}")
    
    # Instantiate agents
    agents = {
        "DataCollector": DataCollector(),
        "FundamentalAnalyzer": FundamentalAnalyzer(),
        "TechnicalAnalyzer": TechnicalAnalyzer(),
        "SentimentAnalyzer": SentimentAnalyzer(),
        "RiskAnalyzer": RiskAnalyzer(),
        "AIResearcher": AIResearcher(),
        "TraderAgent": TraderAgent()
    }
    
    # Run data collection agents and collect outputs
    agent_outputs = {}
    for agent_name, agent in list(agents.items())[:-2]:  # Exclude AIResearcher and TraderAgent
        logger.info(f"Running {agent_name}")
        agent.run(stock_symbol)
        msg = receive_message()
        if msg:
            agent_outputs[agent_name] = msg["message"]
        else:
            logger.warning(f"No output from {agent_name}")
    
    # Re-populate queue for AIResearcher
    logger.info("Re-populating queue for AIResearcher")
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
    
    logger.info(f"Analysis complete for {stock_symbol}, final report: {final_report}")
    return {
        "stock": stock_symbol,
        "report": final_report,
        "agent_outputs": agent_outputs
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)