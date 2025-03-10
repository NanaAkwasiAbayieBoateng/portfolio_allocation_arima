import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import pmdarima as pm



# **Fetch S&P 500 Tickers from Wikipedia**
@st.cache_data
def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url, header=0)[0]
    return sp500_table["Symbol"].tolist()

# **Fetch historical stock data**
def fetch_asset_data(assets, start_date, end_date):
    data = yf.download(assets, start=start_date, end=end_date)['Close']
    return data.dropna()

# **Train AI Model to Predict Future Stock Returns using Auto ARIMA**
def predict_top_stocks(tickers, start_date, end_date, top_n=5):
    stock_predictions = {}

    for ticker in tickers[:50]:  # Limit for efficiency
        try:
            data = yf.download(ticker, start=start_date, end=end_date)['Close']
            returns = data.pct_change().dropna()

            if len(returns) > 60:
                model = pm.auto_arima(returns, seasonal=False, suppress_warnings=True)
                predicted_returns, conf_int = model.predict(n_periods=30, return_conf_int=True)
                stock_predictions[ticker] = np.mean(predicted_returns)
        except Exception as e:
            print(f"Error predicting {ticker}: {e}") #print error for debugging.
            continue

    sorted_tickers = sorted(stock_predictions, key=stock_predictions.get, reverse=True)
    return sorted_tickers[:top_n]

# **Select Assets Based on Risk Tolerance & AI Predictions**
def select_assets(sp500_tickers, start_date, end_date, risk_tolerance):
    selected_assets = ["SPY"]  # Always include S&P 500 index
    top_predicted_stocks = predict_top_stocks(sp500_tickers, start_date, end_date, top_n=5)
    selected_assets.extend(top_predicted_stocks)

    if risk_tolerance >= 70:
        selected_assets += ["BTC-USD", "ETH-USD"]

    return selected_assets

# **Portfolio Optimization**
def calculate_portfolio_stats(weights, returns, cov_matrix):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return portfolio_return, portfolio_volatility

def optimize_portfolio(returns, cov_matrix, min_return=0.02):
    def negative_sharpe(weights):
        portfolio_return, portfolio_volatility = calculate_portfolio_stats(weights, returns, cov_matrix)
        if portfolio_return <= min_return:
            return np.inf
        return -((portfolio_return - 0.02) / portfolio_volatility)

    num_assets = len(returns.columns)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1.0 / num_assets])

    optimized = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized.x

# **Backtesting and Visualization**
def backtest_and_visualize(data, optimal_weights, start_date, end_date):
    portfolio_returns = (data.pct_change().dropna() @ optimal_weights)
    sp500 = fetch_asset_data(["^GSPC"], start_date, end_date)
    sp500_returns = sp500.pct_change().dropna()['^GSPC']

    cumulative_returns = (1 + portfolio_returns).cumprod()
    sp500_cumulative_returns = (1 + sp500_returns).cumprod()

    # Calculate rolling beta
    rolling_beta = portfolio_returns.rolling(window=252).corr(sp500_returns)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Portfolio Cumulative Return', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=sp500_cumulative_returns.index, y=sp500_cumulative_returns, mode='lines', name='S&P 500 Cumulative Return', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, mode='lines', name='Rolling Beta', line=dict(color='green')))

    fig.update_layout(title='Portfolio vs S&P 500 Performance and Rolling Beta', xaxis_title='Date', yaxis_title='Cumulative Return/Beta', hovermode='x unified')
    return fig

# **STREAMLIT APP**
st.title("📈 AI-Driven Portfolio Optimization")

# **User Inputs**
start_date = st.date_input("Select start date:", value=pd.to_datetime("2024-01-01"))
end_date = st.date_input("Select end date:", value=pd.to_datetime("2024-06-30"))
risk_tolerance = st.slider("Select your risk tolerance (0-100):", 0, 100, 50)

if st.button("Run Portfolio Optimization"):
    st.write("🚀 **Fetching S&P 500 Tickers...**")
    sp500_tickers = fetch_sp500_tickers()

    st.write("🤖 **Using AI to select top-performing stocks...**")
    selected_assets = select_assets(sp500_tickers, start_date, end_date, risk_tolerance)
    st.write(f"✅ **Selected Assets:** {selected_assets}")

    st.write("📊 **Fetching historical stock data...**")
    data = fetch_asset_data(selected_assets, start_date, end_date)

    if data.empty:
        st.error("⚠ No stock data found. Try changing the date range or risk tolerance.")
    else:
        st.write("🔢 **Optimizing Portfolio Allocation...**")
        returns = data.pct_change().dropna()
        cov_matrix = returns.cov()
        optimal_weights = optimize_portfolio(returns, cov_matrix)

        st.write("✅ **Optimal Portfolio Allocation:**")
        allocation_df = pd.DataFrame({"Asset": selected_assets, "Weight (%)": optimal_weights * 100})
        st.dataframe(allocation_df)

        # **Backtesting and Visualization**
        st.write("📈 **Portfolio Backtest Results**")
        fig = backtest_and_visualize(data, optimal_weights, start_date, end_date)
        st.plotly_chart(fig)

        st.success("🎯 Portfolio optimization complete!")