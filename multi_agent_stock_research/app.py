import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from stock_analysis import analyze
import traceback
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime

# Streamlit app configuration
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Title and description
st.title("Stock Analysis Dashboard")
st.markdown("Enter a stock symbol to analyze its market data, fundamentals, technicals, sentiment, and risk profile.")

# Input section in sidebar
with st.sidebar:
    st.header("Input")
    stock_symbol = st.text_input("Stock Symbol (e.g., MSFT, TSLA)", value="MSFT").upper()
    analyze_button = st.button("Analyze")

# Function to generate the PDF report
def generate_pdf_report(result):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Stock Analysis Report for {result['stock']}", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Recommendation", styles['Heading2']))
    story.append(Paragraph(result['report'], styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("AI Researcher Report", styles['Heading2']))
    if 'AIResearcher' in result['agent_outputs']:
        story.append(Paragraph(result['agent_outputs']['AIResearcher']['summary'], styles['BodyText']))
    else:
        story.append(Paragraph("AI Researcher report unavailable.", styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Technical Analysis", styles['Heading2']))
    if 'TechnicalAnalyzer' in result['agent_outputs'] and 'error' not in result['agent_outputs']['TechnicalAnalyzer']:
        tech_data = result['agent_outputs']['TechnicalAnalyzer']
        data = [
            ["Metric", "Value"],
            ["50-day SMA", str(tech_data.get('sma50', 'N/A'))],
            ["200-day SMA", str(tech_data.get('sma200', 'N/A'))],
            ["RSI", str(tech_data.get('rsi', 'N/A'))],
            ["MACD", str(tech_data.get('macd', 'N/A'))],
            ["Bollinger Middle", str(tech_data.get('boll_mid', 'N/A'))],
            ["Bollinger Upper", str(tech_data.get('boll_upper', 'N/A'))],
            ["Bollinger Lower", str(tech_data.get('boll_lower', 'N/A'))],
            ["Support", str(tech_data.get('support', 'N/A'))],
            ["Resistance", str(tech_data.get('resistance', 'N/A'))]
        ]
        table = Table(data)
        table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        story.append(table)
    else:
        story.append(Paragraph("Technical analysis unavailable.", styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Fundamental Analysis", styles['Heading2']))
    if 'FundamentalAnalyzer' in result['agent_outputs'] and 'error' not in result['agent_outputs']['FundamentalAnalyzer']:
        fund_data = result['agent_outputs']['FundamentalAnalyzer']
        data = [
            ["Metric", "Value"],
            ["P/E Ratio", str(fund_data.get('pe_ratio', 'N/A'))],
            ["PEG Ratio", str(fund_data.get('peg_ratio', 'N/A'))],
            ["EPS", str(fund_data.get('eps', 'N/A'))],
            ["Market Cap", str(fund_data.get('market_cap', 'N/A'))],
            ["Dividend Yield (%)", str(fund_data.get('dividend_yield', 'N/A'))],
            ["Debt-to-Equity", str(fund_data.get('debt_to_equity', 'N/A'))],
            ["Sector", str(fund_data.get('sector', 'N/A'))]
        ]
        table = Table(data)
        table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        story.append(table)
    else:
        story.append(Paragraph("Fundamental analysis unavailable.", styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Sentiment Analysis", styles['Heading2']))
    if 'SentimentAnalyzer' in result['agent_outputs'] and 'error' not in result['agent_outputs']['SentimentAnalyzer']:
        sent_data = result['agent_outputs']['SentimentAnalyzer']
        data = [
            ["Metric", "Value"],
            ["Sentiment", str(sent_data.get('sentiment', 'N/A'))],
            ["Score", str(sent_data.get('score', 'N/A'))]
        ]
        table = Table(data)
        table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        story.append(table)
    else:
        story.append(Paragraph("Sentiment analysis unavailable.", styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Risk Profile", styles['Heading2']))
    if 'RiskAnalyzer' in result['agent_outputs'] and 'error' not in result['agent_outputs']['RiskAnalyzer']:
        risk_data = result['agent_outputs']['RiskAnalyzer']
        data = [
            ["Metric", "Value"],
            ["Annualized Volatility", str(risk_data.get('annualized_volatility', 'N/A')) + "%"],
            ["Beta", str(risk_data.get('beta', 'N/A'))],
            ["VaR (95%)", str(risk_data.get('var_95', 'N/A')) + "%"],
            ["10-Year Treasury Yield", str(risk_data.get('treasury_yield', 'N/A')) + "%"]
        ]
        table = Table(data)
        table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        story.append(table)
    else:
        story.append(Paragraph("Risk analysis unavailable.", styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# Main content area
if analyze_button:
    with st.spinner("Analyzing..."):
        try:
            result = analyze(stock_symbol)
            pdf = generate_pdf_report(result)

            st.header(f"Analysis for {result['stock']}")
            st.subheader("Recommendation")
            st.write(result['report'])

            st.download_button(
                label="Download PDF Report",
                data=pdf,
                file_name=f"{stock_symbol}_analysis.pdf",
                mime="application/pdf"
            )

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Technicals", "Fundamentals", "Sentiment", "Risk"])

            with tab1:
                st.markdown("### AI Researcher Report")
                if 'AIResearcher' in result['agent_outputs']:
                    st.markdown(result['agent_outputs']['AIResearcher']['summary'])
                else:
                    st.write("AI Researcher report unavailable.")

            with tab2:
                st.subheader("Historical Price Chart")
                try:
                    history = yf.Ticker(stock_symbol).history(period="1y")
                    fig = go.Figure(data=[go.Candlestick(x=history.index,
                                                         open=history['Open'],
                                                         high=history['High'],
                                                         low=history['Low'],
                                                         close=history['Close'])])
                    fig.update_layout(title="1-Year Historical Price", xaxis_title="Date", yaxis_title="Price (USD)")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.write("Unable to fetch historical data.")

                st.subheader("Technical Indicators")
                if 'TechnicalAnalyzer' in result['agent_outputs'] and 'error' not in result['agent_outputs']['TechnicalAnalyzer']:
                    tech_data = result['agent_outputs']['TechnicalAnalyzer']
                    st.write(f"**50-day SMA**: {tech_data.get('sma50', 'N/A')}")
                    st.write(f"**200-day SMA**: {tech_data.get('sma200', 'N/A')}")
                    st.write(f"**RSI**: {tech_data.get('rsi', 'N/A')}")
                    st.write(f"**MACD**: {tech_data.get('macd', 'N/A')}")
                    st.write(f"**Bollinger Middle Band**: {tech_data.get('boll_mid', 'N/A')}")
                    st.write(f"**Bollinger Upper Band**: {tech_data.get('boll_upper', 'N/A')}")
                    st.write(f"**Bollinger Lower Band**: {tech_data.get('boll_lower', 'N/A')}")
                    st.write(f"**Support**: {tech_data.get('support', 'N/A')}")
                    st.write(f"**Resistance**: {tech_data.get('resistance', 'N/A')}")
                else:
                    st.write("Technical analysis unavailable due to an error:", result['agent_outputs'].get('TechnicalAnalyzer', {}).get('error', 'Unknown error'))

            with tab3:
                st.subheader("Fundamental Metrics")
                if 'FundamentalAnalyzer' in result['agent_outputs'] and 'error' not in result['agent_outputs']['FundamentalAnalyzer']:
                    fund_data = result['agent_outputs']['FundamentalAnalyzer']
                    metrics = {
                        "P/E Ratio": fund_data.get('pe_ratio', 'N/A'),
                        "PEG Ratio": fund_data.get('peg_ratio', 'N/A'),
                        "EPS": fund_data.get('eps', 'N/A'),
                        "Market Cap": fund_data.get('market_cap', 'N/A'),
                        "Dividend Yield (%)": fund_data.get('dividend_yield', 'N/A'),
                        "Debt-to-Equity": fund_data.get('debt_to_equity', 'N/A'),
                        "Sector": fund_data.get('sector', 'N/A')
                    }
                    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))
                else:
                    st.write("Fundamental analysis unavailable due to an error:", result['agent_outputs'].get('FundamentalAnalyzer', {}).get('error', 'Unknown error'))

            with tab4:
                st.subheader("Sentiment Analysis")
                if 'SentimentAnalyzer' in result['agent_outputs'] and 'error' not in result['agent_outputs']['SentimentAnalyzer']:
                    sent_data = result['agent_outputs']['SentimentAnalyzer']
                    st.write(f"**Sentiment**: {sent_data.get('sentiment', 'N/A')} (Score: {sent_data.get('score', 'N/A')})")
                else:
                    st.write("Sentiment analysis unavailable due to an error:", result['agent_outputs'].get('SentimentAnalyzer', {}).get('error', 'Unknown error'))

            with tab5:
                st.subheader("Risk Profile")
                if 'RiskAnalyzer' in result['agent_outputs'] and 'error' not in result['agent_outputs']['RiskAnalyzer']:
                    risk_data = result['agent_outputs']['RiskAnalyzer']
                    st.write(f"**Annualized Volatility**: {risk_data.get('annualized_volatility', 'N/A')}%")
                    st.write(f"**Beta**: {risk_data.get('beta', 'N/A')}")
                    st.write(f"**VaR (95%)**: {risk_data.get('var_95', 'N/A')}%")
                    st.write(f"**10-Year Treasury Yield**: {risk_data.get('treasury_yield', 'N/A')}%")
                    summary = result['agent_outputs'].get('AIResearcher', {}).get('summary', '').lower()
                    risk_level = "N/A"
                    if "risk profile is low" in summary:
                        risk_level = "Low"
                    elif "risk profile is moderate" in summary:
                        risk_level = "Moderate"
                    elif "risk profile is high" in summary:
                        risk_level = "High"
                    st.write(f"**Risk Level**: {risk_level}")
                else:
                    st.write("Risk analysis unavailable due to an error:", result['agent_outputs'].get('RiskAnalyzer', {}).get('error', 'Unknown error'))

        except Exception as e:
            st.error(f"Error during analysis: {traceback.format_exc()}")

# Footer
st.markdown("---")
st.write("Powered by Streamlit and OpenAI GPT-4 Turbo")