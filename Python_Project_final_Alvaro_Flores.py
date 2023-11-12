#==============================================================================
# Initiating
#==============================================================================

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
from plotly.subplots import make_subplots
from stocknews import StockNews


#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret

#==============================================================================
# Sidebar
#==============================================================================

# Define custom function for the sidebar
def custom_sidebar():
    st.sidebar.title("MY FINANCIAL DASHBOARD ⭐")
    
    # Get the list of stock tickers from S&P500
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
   
    global ticker # Set this variable as global, so the functions in all of the tabs can read it
    ticker = st.sidebar.selectbox("Ticker", ticker_list)
    st.sidebar.write(f"You selected: {ticker}")
    
    global update_button  # Set this variable as global
    # Create an "Update" button in the sidebar
    update_button = st.sidebar.button("Update Data")
    
    
#==============================================================================
# Tab 1
#==============================================================================


def render_tab1a():
    
    # Creation two columns
    col1, col2 = st.columns(2)
    
                        
    # Display stock information when the button is clicked
    if update_button or ticker != '':
        
        
        stock_info = YFinance(ticker).info
      

        with col1:
            st.write("Previous Close:", str(stock_info["previousClose"]))
            st.write("Open:", str(stock_info["open"]))
            st.write("Bid:", str(stock_info["bid"]))
            st.write("Ask:", str(stock_info["ask"]))
            st.write("Day's Range:", str(stock_info["dayLow"]), "-", str(stock_info["dayHigh"]))
            st.write("52 Week Range:", str(stock_info.get("fiftyTwoWeekLow", "N/A")), " - ", str(stock_info.get("fiftyTwoWeekHigh", "N/A")))
            st.write("Volume:", '{:,}'.format(stock_info.get("volume", "N/A")))
            st.write("Average Volume:", '{:,}'.format(stock_info.get("averageVolume", "N/A")))
            
            
            # Stock Price History Graphic
            
            # Create a dictionary to map duration options to the corresponding yfinance periods
            duration_to_period = {
                 "1M": "1mo",
                 "3M": "3mo",
                 "6M": "6mo",
                 "YTD": "ytd",
                 "1Y": "1y",
                 "3Y": "3y",
                 "5Y": "5y",
                 "MAX": "max",
             }
                      
             # Creation of the Streamlit Title
            st.markdown(f"<h1 style='text-align: left; font-size: 18px;'>{ticker} Stock Price History</h1>", unsafe_allow_html=True)
             
             # Adding a selectbox for choosing the duration
            selected_duration = st.selectbox("Select Duration", list(duration_to_period.keys()))
             
             # Collecion of the corresponding data based on the selected duration
            stock_data = yf.Ticker(ticker).history(period=duration_to_period[selected_duration])
           
             # Creation of a Plotly figure with an area chart
            fig = go.Figure(data=go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', fill='tozeroy', name='Closing Price'))
             
             # Set of the chart title and axis labels
            fig.update_layout(
                 title=f"{ticker} Closing Price Over {selected_duration}",
                 xaxis_title="Date",
                 yaxis_title="Closing Price",
             )
             
             # Display of the interactive area chart
            st.plotly_chart(fig)
        
        with col2:
            st.write("Market Cap:", '{:,}'.format(stock_info.get("marketCap", "N/A")))
            st.write("Beta (5Y Monthly):", str(stock_info.get("beta", "N/A")))
            st.write("PE Ratio (TTM):", str(stock_info.get("trailingPE", "N/A")))
            st.write("EPS (TTM):", str(stock_info.get("trailingEps", "N/A")))
            st.write("Earnings Date:", stock_info.get("earningsdate", "N/A"))
            st.write("Forward Dividend & Yield:", str(stock_info.get("dividendRate", "N/A")), "-", str(stock_info.get("dividendYield", "N/A")))
            st.write("Ex-Dividend Date:", datetime.utcfromtimestamp(stock_info.get("exDividendDate", "N/A")).strftime('%Y-%m-%d'))
            st.write("1-Year Target Estimate:", str(stock_info.get("targetMeanPrice", "N/A")))

 

#==============================================================================
# Tab 2
#==============================================================================

def render_tab2():
    if ticker:
        # Retrieve of historical stock data
        stock_data = yf.Ticker(ticker).history(period="max", interval="1d")
       
        # Select of the date range for showing the stock price
        date_range_start = st.date_input("Start Date:", stock_data.index.min().date())
        date_range_end = st.date_input("End Date:", stock_data.index.max().date())
    
        # Convert of date_range_start and date_range_end to Pandas DateTime objects with the same timezone as stock_data
        date_range_start = pd.to_datetime(date_range_start, utc=True).tz_convert(stock_data.index.tz)
        date_range_end = pd.to_datetime(date_range_end, utc=True).tz_convert(stock_data.index.tz)
    
        # Filter of the stock data based on the selected date range
        filtered_data = stock_data[
            (stock_data.index >= date_range_start) & (stock_data.index <= date_range_end)
        ]
   
        # Select the duration of time
        duration = st.selectbox("Duration", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"])
    
        if duration == "1M":
            filtered_data = filtered_data[-30:]
        elif duration == "3M":
            filtered_data = filtered_data[-90:]
        elif duration == "6M":
            filtered_data = filtered_data[-180:]
        elif duration == "YTD":
            year_start = pd.to_datetime(f'{pd.Timestamp.now().year}-01-01')
            filtered_data = filtered_data[filtered_data.index >= year_start]
        elif duration == "1Y":
            filtered_data = filtered_data[-365:]
        elif duration == "3Y":
            filtered_data = filtered_data[-1095:]
        elif duration == "5Y":
            filtered_data = filtered_data[-1825:]
    
        # Select of the time interval
        time_interval = st.selectbox("Time Interval", ["1D", "1W", "1M", "1Y"])
    
        # Resample of the stock data to the selected time interval
        resampled_data = filtered_data.resample(time_interval).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    
        # Show of the stock price using line or candle plot
        plot_type = st.selectbox("Plot Type", ["Line", "Candle"])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        
        resampled_data_interpolated = resampled_data.interpolate(method='linear')

        if plot_type == "Line":
            fig.add_trace(go.Scatter(x=resampled_data.index, y=resampled_data["Close"], mode='lines', name='Close Price', connectgaps=True), row=1, col=1)
        else:
            fig.add_trace(go.Candlestick(x=resampled_data_interpolated.index, open=resampled_data_interpolated['Open'], high=resampled_data_interpolated['High'], low=resampled_data_interpolated['Low'], close=resampled_data_interpolated['Close']), row=1, col=1)

        fig.add_trace(go.Bar(x=resampled_data.index, y=resampled_data["Volume"], name='Volume'), row=2, col=1)

        # Calculation and plotting of the 50 days Simple Moving Average (MA)
        ma50 = resampled_data["Close"].rolling(window=50, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=ma50.index, y=ma50, mode='lines', name='SMA 50', line=dict(color='red'), showlegend=False), row=1, col=1)

        # Configureation of the layout
        fig.update_layout(title=f'{ticker} Stock Price Chart', xaxis_title='', yaxis_title='Price', xaxis2_title="Date", yaxis2_title="Volume")
        fig.update_yaxes(title_text='Volume', secondary_y=True)
        fig.update_xaxes(rangeslider_visible=False)

        # Display ofthe chart
        st.plotly_chart(fig)
#==============================================================================
# Tab 3
#==============================================================================

def render_tab3():

    st.title("Stock Financials")
    
    # Select of the financial statement type
    statement_type = st.selectbox(
        "Select Financial Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"]
    )
    
    # Select of the period
    period = st.selectbox("Select Period", ["Annual", "Quarterly"])
    
    if ticker:
        try:
            stock = yf.Ticker(ticker)
    
            if statement_type == "Income Statement":
                if period == "Annual":
                    data = stock.financials
                else:
                    data = stock.quarterly_financials
            elif statement_type == "Balance Sheet":
                if period == "Annual":
                    data = stock.balance_sheet
                else:
                    data = stock.quarterly_balance_sheet
            else:
                if period == "Annual":
                    data = stock.cashflow
                else:
                    data = stock.quarterly_cashflow
    
            if not data.empty:
                st.title(f"{ticker} {statement_type} ({period})")
                st.write(data)
    
            else:
                st.error(f"No data available for {ticker} {statement_type} ({period}).")
    
        except Exception as e:
            st.error(f"Error: {e}")



#==============================================================================
# Tab 4
#==============================================================================


def render_tab4():
    
    # Get the Apple stock price from Yahoo finance
    stock_price = yf.Ticker(ticker).history(start='2022-11-01', end='2023-10-31')

        
    # Take the close price
    close_price = stock_price['Close']
        
    # Plot close stock pric
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(close_price)
 
    
    # The returns ((today price - yesterday price) / yesterday price)
    daily_return = close_price.pct_change()
  
    
    # The volatility (high value, high risk)
    daily_volatility = np.std(daily_return)
    
    
    # Select the simulations
    simulations = st.selectbox(
        "Select Number of Simulations", [200,500,1000])
    
    # Select the time_horizon
    time_horizon = st.selectbox(
        "Select The Time Horizon", [30,60,90])
    

    # Run the simulation
    simulation_df = pd.DataFrame()
    
    for i in range(simulations):
        # The list to store the next stock price
        next_price = []
    
        # Create the next stock price
        last_price = close_price[-1]
    
        for j in range(time_horizon):
            # Generate the random percentage change around the mean (0) and std (daily_volatility)
            future_return = np.random.normal(0, daily_volatility)
    
            # Generate the random future price
            future_price = last_price * (1 + future_return)
    
            # Save the price and go next
            next_price.append(future_price)
            last_price = future_price
    
        # Store the result of the simulation
        next_price_df = pd.Series(next_price).rename('sim' + str(i))
        simulation_df = pd.concat([simulation_df, next_price_df], axis=1)
    
    
    # Create a Streamlit figure
    st.write("Monte Carlo Simulation Plot:")
    fig = plt.figure(figsize=(15, 10))
    plt.plot(simulation_df)
    plt.axhline(y=close_price[-1], color='red')
    plt.title(f"Monte Carlo simulation for {ticker} stock price in the next {time_horizon} days")
    plt.xlabel('Day')
    plt.ylabel('Price')
    legend_text = ['Current stock price is: ' + str(np.round(close_price[-1], 2))]
    plt.legend(legend_text)
    plt.gca().get_legend().get_lines()[0].set_color('red')
    st.pyplot(fig)


#==============================================================================
# Tab 5
#==============================================================================

def render_tab5():
    
            
    st.header(f'News of {ticker}')
    stocks = [ticker]
    sn = StockNews(stocks, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
            st.subheader(f'News {i+1}')
            st.write(df_news['published'][i])
            st.write(df_news['title'][i])
            st.write(df_news['summary'][i])
            title_sentiment = df_news['sentiment_title'][i]
            st.write(f'Title Sentiment {title_sentiment}')
            news_sentiment = df_news['sentiment_summary'][i]
            st.write(f'News Sentiment {news_sentiment}')

#==============================================================================
# Main body
#==============================================================================
      

# Call the custom_sidebar function to display the sidebar
custom_sidebar()

# Render the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stock Summary", "Price Chart", "Financials","Monte Carlo Simulation", "News and Sentiment Analysis"])

with tab1:
    render_tab1a()
  
    
with tab2:
    render_tab2()

with tab3:
    render_tab3()

with tab4:
    render_tab4()
    
with tab5:
    render_tab5()
    
    
# Customize the dashboard with CSS
st.markdown(
    """
    <style>
        .stApp {
            background: #F0F8FF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    
###############################################################################
# END
###############################################################################
