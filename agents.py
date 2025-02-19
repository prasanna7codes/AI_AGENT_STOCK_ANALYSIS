import streamlit as st
import pandas as pd
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()

# Function to get stock symbol
def get_company_symbol(company: str) -> str:
    symbols = {
        # US Tech Giants
        'NVIDIA': 'NVDA', 'APPLE': 'AAPL', 'GOOGLE': 'GOOGL', 'MICROSOFT': 'MSFT',
        'TESLA': 'TSLA', 'AMAZON': 'AMZN', 'META': 'META', 'NETFLIX': 'NFLX',

        # Indian Market
        'TCS': 'TCS.NS', 'RELIANCE': 'RELIANCE.NS', 'INFOSYS': 'INFY.NS',
        'WIPRO': 'WIPRO.NS', 'HDFC': 'HDFCBANK.NS', 'TATAMOTORS': 'TATAMOTORS.NS',
        'ICICIBANK': 'ICICIBANK.NS', 'SBIN': 'SBIN.NS', 'L&T': 'LT.NS', 'BAJAJ FINANCE': 'BAJFINANCE.NS',
        'HUL': 'HINDUNILVR.NS', 'ITC': 'ITC.NS', 'MARUTI': 'MARUTI.NS', 'ASIAN PAINTS': 'ASIANPAINT.NS',

        # US Financials & Industrials
        'JPMORGAN': 'JPM', 'GOLDMAN SACHS': 'GS', 'BANK OF AMERICA': 'BAC',
        'WELLS FARGO': 'WFC', 'BERKSHIRE HATHAWAY': 'BRK-B',
        'BOEING': 'BA', 'CATERPILLAR': 'CAT',

        # Energy & Automotive
        'EXXON MOBIL': 'XOM', 'CHEVRON': 'CVX', 'FORD': 'F',
        'GENERAL MOTORS': 'GM', 'TOYOTA': 'TM', 'HONDA': 'HMC',

        # European Giants
        'SAP': 'SAP', 'SIEMENS': 'SIEGY', 'LOREAL': 'OR.PA',
        'UNILEVER': 'UL', 'NESTLE': 'NESN.SW', 'ROCHE': 'ROG.SW',

        # Chinese & Asian Market
        'ALIBABA': 'BABA', 'TENCENT': '0700.HK', 'BAIDU': 'BIDU',
        'SAMSUNG': '005930.KQ', 'TSMC': 'TSM',

        # Cryptocurrencies (if needed)
        'BITCOIN': 'BTC-USD', 'ETHEREUM': 'ETH-USD'
    }
    return symbols.get(company.upper(), "Unknown")


# Web agent for fetching news
web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama-3.3-70b-specdec"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Finance agent for fetching stock data
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama-3.3-70b-specdec"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data.", "Use markdown for clear formatting."],
    show_tool_calls=True,
    markdown=True,
)

# Combined agent team
agent_team = Agent(
    model=Groq(id="llama-3.3-70b-specdec"),
    team=[web_agent, finance_agent],
    instructions=["Provide separate sections for news and stock data.", "Use tables when appropriate."],
    show_tool_calls=True,
    markdown=True,
)

# Function to extract a markdown table
def extract_markdown_table(text):
    """Extracts markdown tables from the response and converts to DataFrame."""
    table_lines = []
    capturing = False

    for line in text.split("\n"):
        if "|" in line:
            table_lines.append(line.strip())
            capturing = True
        elif capturing:
            break  # Stop capturing once table ends

    if len(table_lines) < 3:
        return None

    header = [col.strip() for col in table_lines[0].split("|")[1:-1]]
    data = [[col.strip() for col in row.split("|")[1:-1]] for row in table_lines[2:]]

    return pd.DataFrame(data, columns=header)

# Streamlit UI
st.title("📊 Company News & Stock Analysis")

company_name = st.text_input("Enter Company Name (e.g., TCS, NVIDIA)", "TCS")

if st.button("Get Data"):
    symbol = get_company_symbol(company_name)
    
    with st.spinner("Fetching latest news and stock data..."):
        # Fetch news and stock data
        response = agent_team.run(f"Get the latest news and stock details for {symbol}")
        response_text = response.content  # ✅ Make sure we're correctly getting the content

        st.subheader(f"📰 Latest News for {company_name} ({symbol})")
        
        # ✅ Extract news section if available
        news_section_start = response_text.find("### Latest News")
        stock_section_start = response_text.find("### Stock Data")

        if news_section_start != -1:
            if stock_section_start != -1:
                news_text = response_text[news_section_start:stock_section_start].strip()
            else:
                news_text = response_text[news_section_start:].strip()

            st.markdown(news_text)  # ✅ Display the news as markdown
        else:
            st.warning("⚠️ No news data found.")

        # ✅ Extract stock data table
        stock_data = extract_markdown_table(response_text)
        if stock_data is not None:
            st.subheader(f"📉 Stock Data for {company_name} ({symbol})")
            st.table(stock_data)
        else:
            st.warning("⚠️ No stock data found.")
