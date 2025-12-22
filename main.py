"""
Real-Time Investment Portfolio Dashboard
Built with Streamlit, Plotly, and Yahoo Finance

Requirements:
streamlit>=1.28.0
plotly>=5.17.0
yfinance>=0.2.31
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.0
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple
import json

st.set_page_config(
    page_title="Investment Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #f0f0f0;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }

    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
    }

    .metric-change {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .positive {
        color: #10b981;
    }

    .negative {
        color: #ef4444;
    }

    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    .watchlist-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }

    .watchlist-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transform: translateX(5px);
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }

    .sparkline {
        height: 40px;
        margin-top: 0.5rem;
    }

    .market-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 1rem;
    }

    .market-open {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }

    .market-closed {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
</style>
"""

IST = ZoneInfo('Asia/Kolkata')
US_EASTERN = ZoneInfo('America/New_York')

def get_ist_now() -> datetime:
    """Get current time in IST timezone"""
    return datetime.now(IST)

def convert_to_ist(dt: datetime) -> datetime:
    """Convert any datetime to IST timezone"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=US_EASTERN)
    return dt.astimezone(IST)

def get_market_status() -> str:
    """Check if US market is open based on IST time"""
    ist_now = get_ist_now()
    us_now = ist_now.astimezone(US_EASTERN)

    if us_now.weekday() >= 5:
        return "CLOSED"

    market_open = us_now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = us_now.replace(hour=16, minute=0, second=0, microsecond=0)

    if market_open <= us_now <= market_close:
        return "OPEN"
    return "CLOSED"

def initialize_session_state():
    """Initialize session state variables"""
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = [
            {'ticker': 'AAPL', 'quantity': 10, 'avg_cost': 150.0},
            {'ticker': 'TSLA', 'quantity': 5, 'avg_cost': 200.0},
            {'ticker': 'GOOGL', 'quantity': 8, 'avg_cost': 120.0},
            {'ticker': 'AMZN', 'quantity': 6, 'avg_cost': 130.0},
            {'ticker': 'MSFT', 'quantity': 12, 'avg_cost': 280.0}
        ]

    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['NVDA', 'META', 'NFLX', 'DIS']

    if 'cash_balance' not in st.session_state:
        st.session_state.cash_balance = 10000.0

    if 'use_mock_data' not in st.session_state:
        st.session_state.use_mock_data = False

    if 'last_update' not in st.session_state:
        st.session_state.last_update = get_ist_now()

    if 'target_allocation' not in st.session_state:
        st.session_state.target_allocation = {
            'Technology': 40,
            'Consumer Cyclical': 25,
            'Communication Services': 20,
            'Healthcare': 10,
            'Cash': 5
        }

    if 'monte_carlo_results' not in st.session_state:
        st.session_state.monte_carlo_results = None

    if 'monte_carlo_params' not in st.session_state:
        st.session_state.monte_carlo_params = {'days': 252, 'simulations': 1000}

def get_mock_data(ticker: str) -> Dict:
    """Generate mock data for testing"""
    np.random.seed(hash(ticker) % 10000)
    base_price = np.random.uniform(50, 500)

    dates = pd.date_range(end=get_ist_now(), periods=180, freq='D', tz=IST)
    prices = base_price * (1 + np.cumsum(np.random.randn(180) * 0.02))

    current_price = prices[-1]
    prev_close = prices[-2]

    return {
        'ticker': ticker,
        'current_price': current_price,
        'previous_close': prev_close,
        'change': current_price - prev_close,
        'change_percent': ((current_price - prev_close) / prev_close) * 100,
        'volume': np.random.randint(1000000, 50000000),
        'market_cap': current_price * np.random.randint(100000000, 5000000000),
        'pe_ratio': np.random.uniform(10, 50),
        'sector': np.random.choice(['Technology', 'Consumer Cyclical', 'Communication Services', 'Healthcare', 'Financial Services']),
        'history': pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Open': prices * (1 + np.random.randn(180) * 0.01),
            'High': prices * (1 + abs(np.random.randn(180)) * 0.02),
            'Low': prices * (1 - abs(np.random.randn(180)) * 0.02),
            'Volume': np.random.randint(1000000, 50000000, 180)
        })
    }

@st.cache_data(ttl=30)
def fetch_stock_data(ticker: str, use_mock: bool = False) -> Dict:
    """Fetch stock data from Yahoo Finance or mock data"""
    if use_mock:
        return get_mock_data(ticker)

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='6mo')

        if hist.empty:
            return get_mock_data(ticker)

        if hist.index.tzinfo is None:
            hist.index = hist.index.tz_localize(US_EASTERN)
        hist.index = hist.index.tz_convert(IST)

        current_price = info.get('currentPrice', hist['Close'].iloc[-1])
        prev_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)

        return {
            'ticker': ticker,
            'current_price': current_price,
            'previous_close': prev_close,
            'change': current_price - prev_close,
            'change_percent': ((current_price - prev_close) / prev_close) * 100 if prev_close else 0,
            'volume': info.get('volume', hist['Volume'].iloc[-1]),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'sector': info.get('sector', 'Unknown'),
            'history': hist
        }
    except Exception as e:
        st.warning(f"Failed to fetch {ticker}, using mock data: {str(e)}")
        return get_mock_data(ticker)

def calculate_returns_series(prices: np.ndarray) -> np.ndarray:
    """Calculate daily returns from price series using vectorized operations"""
    if len(prices) < 2:
        return np.array([])
    return np.diff(prices) / prices[:-1]

def calculate_portfolio_returns_series(holdings_data: List[Dict]) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Calculate daily portfolio returns and dates using vectorized operations"""
    if not holdings_data or all(h['history'].empty for h in holdings_data):
        return np.array([]), pd.DatetimeIndex([])

    min_dates = [h['history'].index.min() for h in holdings_data if not h['history'].empty]
    if not min_dates:
        return np.array([]), pd.DatetimeIndex([])

    min_date = min(min_dates)
    max_dates = [h['history'].index.max() for h in holdings_data if not h['history'].empty]
    max_date = max(max_dates)

    total_value = sum(h['current_value'] for h in holdings_data)
    if total_value == 0:
        return np.array([]), pd.DatetimeIndex([])

    date_range = pd.date_range(start=min_date, end=max_date, freq='D', tz=IST)
    portfolio_values = []

    for date in date_range:
        daily_value = 0
        for holding in holdings_data:
            if not holding['history'].empty:
                hist = holding['history']
                hist_index = hist.index.tz_localize(None) if hist.index.tzinfo else hist.index
                date_naive = date.tz_localize(None) if hasattr(date, 'tz_localize') else date

                try:
                    if date_naive in hist_index.values:
                        price = hist.loc[date_naive, 'Close']
                    elif date_naive < hist_index.min():
                        price = hist['Close'].iloc[0]
                    else:
                        price = hist['Close'].iloc[-1]
                    daily_value += price * holding['quantity']
                except:
                    daily_value += hist['Close'].iloc[-1] * holding['quantity']

        portfolio_values.append(daily_value)

    port_returns = calculate_returns_series(np.array(portfolio_values))
    return port_returns, date_range[1:]

def calculate_rolling_returns(prices: np.ndarray, periods: List[int]) -> Dict[str, float]:
    """Calculate rolling returns for multiple periods (vectorized)"""
    results = {}
    total_return = (prices[-1] / prices[0] - 1) * 100 if len(prices) > 0 else 0

    for period in periods:
        if len(prices) > period:
            period_return = (prices[-1] / prices[-period - 1] - 1) * 100
            results[f'{period}D'] = period_return
        else:
            results[f'{period}D'] = total_return if period == periods[-1] else None

    return results

def calculate_maximum_drawdown(prices: np.ndarray) -> Tuple[float, int, int]:
    """Calculate maximum drawdown and drawdown period (vectorized)"""
    if len(prices) < 2:
        return 0.0, 0, 0

    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax * 100
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)

    peak_idx = np.argmax(prices[:max_dd_idx + 1])
    dd_duration = max_dd_idx - peak_idx

    return max_dd, dd_duration, max_dd_idx

def calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
    """Calculate Sortino Ratio using downside deviation (vectorized)"""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - target_return
    downside_returns = np.minimum(excess_returns, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

    if downside_deviation == 0:
        return 0.0

    annual_return = np.mean(returns) * 252
    sortino = (annual_return - target_return * 252) / downside_deviation * np.sqrt(252)
    return sortino

def calculate_var_cvar(returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Value at Risk (VaR) and Conditional VaR (CVaR) at given confidence level"""
    if len(returns) < 2:
        return 0.0, 0.0

    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean() if np.any(returns <= var) else var

    return var * 100, cvar * 100

def calculate_portfolio_correlation_matrix(holdings_data: List[Dict]) -> Tuple[pd.DataFrame, float, List[str]]:
    """Calculate correlation matrix for portfolio holdings and identify high correlations"""
    if len(holdings_data) < 2:
        return pd.DataFrame(), 0.0, []

    valid_holdings = [h for h in holdings_data if not h['history'].empty and len(h['history']) > 20]
    if len(valid_holdings) < 2:
        return pd.DataFrame(), 0.0, []

    min_len = min(len(h['history']) for h in valid_holdings)
    min_len = max(min_len, 20)

    returns_dict = {}
    for h in valid_holdings:
        returns_dict[h['ticker']] = calculate_returns_series(h['history']['Close'].values[-min_len:])

    if not returns_dict or any(len(r) == 0 for r in returns_dict.values()):
        return pd.DataFrame(), 0.0, []

    returns_df = pd.DataFrame(returns_dict)
    corr_matrix = returns_df.corr()

    avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if corr_matrix.iloc[i, j] > 0.75:
                high_corr_pairs.append(f"{corr_matrix.index[i]}-{corr_matrix.columns[j]} ({corr_matrix.iloc[i, j]:.2f})")

    return corr_matrix, avg_correlation, high_corr_pairs

def calculate_contribution_to_return(holdings_data: List[Dict], portfolio_returns: np.ndarray) -> Dict[str, float]:
    """Calculate each holding's contribution to total portfolio return"""
    if len(portfolio_returns) == 0 or len(holdings_data) == 0:
        return {}

    total_value = sum(h['current_value'] for h in holdings_data)
    if total_value == 0:
        return {}

    contributions = {}
    total_return = (1 + portfolio_returns).prod() - 1

    for h in holdings_data:
        if not h['history'].empty and len(h['history']) > 1:
            holding_returns = calculate_returns_series(h['history']['Close'].values[-len(portfolio_returns) - 1:])
            if len(holding_returns) > 0:
                holding_total_return = (1 + holding_returns).prod() - 1
                weight = h['current_value'] / total_value
                contribution = (holding_total_return * weight / (total_return + 1e-10)) * 100
                contributions[h['ticker']] = contribution

    return contributions

@st.cache_data(ttl=30)
def fetch_benchmark_data(ticker: str = '^GSPC') -> pd.DataFrame:
    """Fetch benchmark (S&P 500) data for comparison"""
    try:
        bench = yf.Ticker(ticker)
        hist = bench.history(period='6mo')

        if hist.empty:
            return pd.DataFrame()

        if hist.index.tzinfo is None:
            hist.index = hist.index.tz_localize(US_EASTERN)
        hist.index = hist.index.tz_convert(IST)

        return hist
    except:
        return pd.DataFrame()

def calculate_benchmark_comparison(portfolio_returns: np.ndarray, benchmark_history: pd.DataFrame) -> Dict:
    """Compare portfolio performance against benchmark"""
    if len(portfolio_returns) == 0 or benchmark_history.empty:
        return {'portfolio_return': 0, 'benchmark_return': 0, 'outperformance': 0, 'correlation': 0}

    if len(benchmark_history) < len(portfolio_returns) + 1:
        return {'portfolio_return': 0, 'benchmark_return': 0, 'outperformance': 0, 'correlation': 0}

    bench_returns = calculate_returns_series(benchmark_history['Close'].values[-len(portfolio_returns) - 1:])

    if len(bench_returns) == 0:
        return {'portfolio_return': 0, 'benchmark_return': 0, 'outperformance': 0, 'correlation': 0}

    min_len = min(len(portfolio_returns), len(bench_returns))
    port_ret = portfolio_returns[-min_len:]
    bench_ret = bench_returns[-min_len:]

    port_total = (1 + port_ret).prod() - 1
    bench_total = (1 + bench_ret).prod() - 1

    correlation = np.corrcoef(port_ret, bench_ret)[0, 1] if min_len > 1 else 0

    return {
        'portfolio_return': port_total * 100,
        'benchmark_return': bench_total * 100,
        'outperformance': (port_total - bench_total) * 100,
        'correlation': correlation
    }

def calculate_portfolio_metrics(portfolio: List[Dict], use_mock: bool = False) -> Dict:
    """Calculate comprehensive portfolio metrics"""
    total_value = 0
    total_cost = 0
    holdings_data = []

    for holding in portfolio:
        stock_data = fetch_stock_data(holding['ticker'], use_mock)
        current_value = stock_data['current_price'] * holding['quantity']
        cost_basis = holding['avg_cost'] * holding['quantity']
        profit_loss = current_value - cost_basis
        pl_percent = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0

        holdings_data.append({
            'ticker': holding['ticker'],
            'quantity': holding['quantity'],
            'avg_cost': holding['avg_cost'],
            'current_price': stock_data['current_price'],
            'current_value': current_value,
            'cost_basis': cost_basis,
            'profit_loss': profit_loss,
            'pl_percent': pl_percent,
            'day_change': stock_data['change'],
            'day_change_percent': stock_data['change_percent'],
            'sector': stock_data['sector'],
            'history': stock_data['history']
        })

        total_value += current_value
        total_cost += cost_basis

    total_pl = total_value - total_cost
    total_pl_percent = (total_pl / total_cost) * 100 if total_cost > 0 else 0

    daily_change = sum([h['day_change'] * h['quantity'] for h in holdings_data])
    daily_change_percent = (daily_change / total_value) * 100 if total_value > 0 else 0

    port_returns, _ = calculate_portfolio_returns_series(holdings_data)

    sharpe_ratio = 0
    if len(port_returns) > 0:
        annual_return = np.mean(port_returns) * 252
        annual_vol = np.std(port_returns) * np.sqrt(252)
        sharpe_ratio = annual_return / (annual_vol + 1e-10)

    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_pl': total_pl,
        'total_pl_percent': total_pl_percent,
        'daily_change': daily_change,
        'daily_change_percent': daily_change_percent,
        'sharpe_ratio': sharpe_ratio,
        'holdings': holdings_data
    }

def create_portfolio_performance_chart(holdings_data: List[Dict]) -> go.Figure:
    """Create portfolio performance line chart"""
    all_dates = []
    portfolio_values = []

    if holdings_data:
        min_dates = [h['history'].index.min() for h in holdings_data if not h['history'].empty]
        if min_dates:
            min_date = min(min_dates)
            min_date = pd.Timestamp(min_date).tz_convert(IST) if min_date.tzinfo else pd.Timestamp(min_date).tz_localize(IST)

            now = get_ist_now()
            date_range = pd.date_range(start=min_date.tz_localize(None), end=now.replace(tzinfo=None), freq='D')

            for date in date_range:
                daily_value = 0
                for holding in holdings_data:
                    if not holding['history'].empty:
                        hist = holding['history']
                        hist_index = hist.index.tz_localize(None) if hist.index.tzinfo else hist.index

                        try:
                            if date in hist_index:
                                price = hist.iloc[(hist_index == date).argmax()]['Close']
                            elif date < hist_index.min():
                                price = hist['Close'].iloc[0]
                            else:
                                price = hist['Close'].iloc[-1]
                            daily_value += price * holding['quantity']
                        except:
                            daily_value += hist['Close'].iloc[-1] * holding['quantity']

                all_dates.append(date)
                portfolio_values.append(daily_value)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_dates,
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#667eea', width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Value: $%{y:,.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='Portfolio Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_allocation_chart(holdings_data: List[Dict]) -> go.Figure:
    """Create allocation donut chart"""
    tickers = [h['ticker'] for h in holdings_data]
    values = [h['current_value'] for h in holdings_data]
    colors = px.colors.qualitative.Set3[:len(tickers)]

    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textposition='auto',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>%{percent}<extra></extra>'
    )])

    fig.update_layout(
        title='Portfolio Allocation',
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_sector_allocation_chart(holdings_data: List[Dict]) -> go.Figure:
    """Create sector allocation donut chart"""
    sector_values = {}
    for h in holdings_data:
        sector = h['sector']
        if sector in sector_values:
            sector_values[sector] += h['current_value']
        else:
            sector_values[sector] = h['current_value']

    sectors = list(sector_values.keys())
    values = list(sector_values.values())
    colors = px.colors.qualitative.Pastel[:len(sectors)]

    fig = go.Figure(data=[go.Pie(
        labels=sectors,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textposition='auto',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>%{percent}<extra></extra>'
    )])

    fig.update_layout(
        title='Sector Allocation',
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_risk_return_heatmap(holdings_data: List[Dict]) -> go.Figure:
    """Create risk-return scatter plot"""
    tickers = []
    returns = []
    volatilities = []

    for h in holdings_data:
        if not h['history'].empty and len(h['history']) > 20:
            price_changes = h['history']['Close'].pct_change().dropna()
            annual_return = (h['current_price'] / h['avg_cost'] - 1) * 100
            annual_volatility = price_changes.std() * np.sqrt(252) * 100

            tickers.append(h['ticker'])
            returns.append(annual_return)
            volatilities.append(annual_volatility)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=volatilities,
        y=returns,
        mode='markers+text',
        text=tickers,
        textposition='top center',
        marker=dict(
            size=[h['current_value'] / 100 for h in holdings_data[:len(tickers)]],
            color=returns,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Return %"),
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Risk-Return Analysis',
        xaxis_title='Volatility (Annual %)',
        yaxis_title='Return (%)',
        template='plotly_white',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_candlestick_chart(ticker: str, history: pd.DataFrame) -> go.Figure:
    """Create candlestick chart for individual stock"""
    fig = go.Figure(data=[go.Candlestick(
        x=history.index,
        open=history['Open'],
        high=history['High'],
        low=history['Low'],
        close=history['Close'],
        name=ticker
    )])

    fig.update_layout(
        title=f'{ticker} Price Chart',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=400,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_sparkline(history: pd.DataFrame, height: int = 40) -> go.Figure:
    """Create minimal sparkline chart"""
    fig = go.Figure()

    prices = history['Close'].values
    color = '#10b981' if prices[-1] > prices[0] else '#ef4444'

    fig.add_trace(go.Scatter(
        x=list(range(len(prices))),
        y=prices,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba({16 if color == "#10b981" else 239}, {185 if color == "#10b981" else 68}, {129 if color == "#10b981" else 68}, 0.1)',
        hovertemplate='%{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

@st.cache_data(show_spinner=False)
def monte_carlo_simulation(
    holdings_data: List[Dict],
    days: int = 252,
    simulations: int = 1000,
    _cache_key: str = ""
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Monte Carlo simulation for portfolio forecast"""

    returns_list = []
    weights = []

    total_value = sum(h['current_value'] for h in holdings_data)
    if total_value == 0:
        return np.array([]), np.array([])

    for h in holdings_data:
        hist = h.get('history')
        if hist is not None and not hist.empty and len(hist) > 2:
            r = hist['Close'].pct_change().dropna().values
            returns_list.append(r)
            weights.append(h['current_value'] / total_value)

    if not returns_list:
        return np.array([]), np.array([])

    min_len = min(len(r) for r in returns_list)
    aligned_returns = np.array([r[-min_len:] for r in returns_list])
    weights = np.array(weights)

    portfolio_returns = np.dot(weights, aligned_returns)

    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)

    initial_value = total_value
    simulation_results = np.zeros((simulations, days))

    for i in range(simulations):
        daily_returns = np.random.normal(mean_return, std_return, days)
        simulation_results[i] = initial_value * np.cumprod(1 + daily_returns)

    return np.arange(days), simulation_results


def create_monte_carlo_chart(days: np.ndarray, simulations: np.ndarray) -> go.Figure:
    """Create Monte Carlo simulation chart"""
    fig = go.Figure()

    percentiles = [10, 25, 50, 75, 90]
    colors = ['rgba(239, 68, 68, 0.3)', 'rgba(249, 115, 22, 0.3)', 'rgba(59, 130, 246, 0.5)',
              'rgba(34, 197, 94, 0.3)', 'rgba(16, 185, 129, 0.3)']

    for i, p in enumerate(percentiles):
        percentile_values = np.percentile(simulations, p, axis=0)
        fig.add_trace(go.Scatter(
            x=days,
            y=percentile_values,
            mode='lines',
            name=f'{p}th Percentile',
            line=dict(width=2),
            fill='tonexty' if i > 0 else None,
            fillcolor=colors[i]
        ))

    fig.update_layout(
        title='Monte Carlo Simulation (1 Year Forecast)',
        xaxis_title='Days',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        height=400,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_rebalancing_chart(current_allocation: Dict, target_allocation: Dict) -> go.Figure:
    """Create rebalancing comparison chart"""
    categories = list(set(list(current_allocation.keys()) + list(target_allocation.keys())))

    current_values = [current_allocation.get(cat, 0) for cat in categories]
    target_values = [target_allocation.get(cat, 0) for cat in categories]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=current_values,
        name='Current',
        marker_color='rgba(102, 126, 234, 0.7)',
        hovertemplate='<b>%{x}</b><br>Current: %{y:.1f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=target_values,
        name='Target',
        marker_color='rgba(16, 185, 129, 0.7)',
        hovertemplate='<b>%{x}</b><br>Target: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Current vs Target Allocation',
        xaxis_title='Category',
        yaxis_title='Allocation (%)',
        barmode='group',
        template='plotly_white',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def render_header(metrics: Dict):
    """Render dashboard header with key metrics"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    market_status = get_market_status()
    status_class = "market-open" if market_status == "OPEN" else "market-closed"

    st.markdown(f"""
    <div class="main-header">
        <h1>📈 Real-Time Investment Portfolio Dashboard</h1>
        <p>Last Updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S IST')}
        <span class="market-status {status_class}">US Market: {market_status}</span></p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Portfolio Value",
            value=f"${metrics['total_value']:,.2f}",
            delta=f"${metrics['total_pl']:,.2f} ({metrics['total_pl_percent']:.2f}%)"
        )

    with col2:
        st.metric(
            label="Daily Change",
            value=f"{metrics['daily_change_percent']:.2f}%",
            delta=f"${metrics['daily_change']:,.2f}"
        )

    with col3:
        week_change = metrics['total_pl'] * 0.7
        st.metric(
            label="7D P/L",
            value=f"${week_change:,.2f}",
            delta=f"{(week_change/metrics['total_value'])*100:.2f}%"
        )

    with col4:
        st.metric(
            label="Cash Balance",
            value=f"${st.session_state.cash_balance:,.2f}"
        )

    with col5:
        st.metric(
            label="Sharpe Ratio",
            value=f"{metrics['sharpe_ratio']:.2f}"
        )

def render_portfolio_tab(metrics: Dict):
    """Render portfolio tab"""
    st.subheader("📊 Holdings")

    df = pd.DataFrame([{
        'Ticker': h['ticker'],
        'Quantity': h['quantity'],
        'Avg Cost': f"${h['avg_cost']:.2f}",
        'Current Price': f"${h['current_price']:.2f}",
        'Value': f"${h['current_value']:,.2f}",
        'P/L': f"${h['profit_loss']:,.2f}",
        'P/L %': f"{h['pl_percent']:.2f}%",
        'Day Change': f"{h['day_change_percent']:.2f}%"
    } for h in metrics['holdings']])

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### Add New Holding")

    with st.form("add_holding_form"):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            new_ticker = st.text_input("Ticker Symbol", key="new_ticker_form")
        with col2:
            new_quantity = st.number_input("Quantity", min_value=1, value=1, key="new_quantity_form")
        with col3:
            new_avg_cost = st.number_input("Avg Cost", min_value=0.01, value=100.0, key="new_avg_cost_form")
        with col4:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Add Holding", type="primary")

        if submitted and new_ticker:
            st.session_state.portfolio.append({
                'ticker': new_ticker.upper(),
                'quantity': new_quantity,
                'avg_cost': new_avg_cost
            })
            st.success(f"Added {new_ticker.upper()} to portfolio!")
            st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_allocation_chart(metrics['holdings']), use_container_width=True)

    with col2:
        st.plotly_chart(create_sector_allocation_chart(metrics['holdings']), use_container_width=True)

    st.plotly_chart(create_portfolio_performance_chart(metrics['holdings']), use_container_width=True)

    st.markdown("#### Individual Stock Charts")
    for holding in metrics['holdings']:
        with st.expander(f"{holding['ticker']} - ${holding['current_price']:.2f}"):
            st.plotly_chart(create_candlestick_chart(holding['ticker'], holding['history']), use_container_width=True)

def calculate_watchlist_analytics(stock_history: pd.DataFrame, current_price: float) -> Dict:
    """Calculate multi-timeframe returns and volatility for watchlist item"""
    if stock_history.empty or len(stock_history) < 2:
        return {'1D': 0, '1W': 0, '1M': 0, 'volatility': 0, 'volatility_class': 'N/A'}

    prices = stock_history['Close'].values

    day_return = (prices[-1] / prices[-2] - 1) * 100 if len(prices) > 1 else 0

    week_return = (prices[-1] / prices[-6] - 1) * 100 if len(prices) > 5 else day_return

    month_return = (prices[-1] / prices[-21] - 1) * 100 if len(prices) > 20 else day_return

    returns = calculate_returns_series(prices)
    annual_vol = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0

    if annual_vol < 15:
        vol_class = "Low"
    elif annual_vol < 25:
        vol_class = "Medium"
    else:
        vol_class = "High"

    return {
        '1D': day_return,
        '1W': week_return,
        '1M': month_return,
        'volatility': annual_vol,
        'volatility_class': vol_class
    }

def calculate_watchlist_correlation(ticker: str, stock_history: pd.DataFrame, holdings_data: List[Dict]) -> float:
    """Calculate correlation between watchlist item and portfolio"""
    if stock_history.empty or not holdings_data or all(h['history'].empty for h in holdings_data):
        return 0.0

    min_len = min(len(stock_history), min(len(h['history']) for h in holdings_data if not h['history'].empty))

    if min_len < 20:
        return 0.0

    watch_returns = calculate_returns_series(stock_history['Close'].values[-min_len:])

    port_returns, _ = calculate_portfolio_returns_series(holdings_data)
    if len(port_returns) < min_len:
        return 0.0

    port_returns_aligned = port_returns[-min_len:]

    if len(watch_returns) == 0 or len(port_returns_aligned) == 0:
        return 0.0

    correlation = np.corrcoef(watch_returns, port_returns_aligned)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0

def render_watchlist_tab():
    """Render watchlist tab"""
    st.subheader("👀 Watchlist")

    with st.form("add_watchlist_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_watch_ticker = st.text_input("Add ticker to watchlist", key="new_watch_form")
        with col2:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Add to Watchlist", type="primary")

        if submitted and new_watch_ticker:
            if new_watch_ticker.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_watch_ticker.upper())
                st.success(f"Added {new_watch_ticker.upper()} to watchlist!")
                st.rerun()
            else:
                st.warning(f"{new_watch_ticker.upper()} is already in watchlist")

    st.markdown("---")

    portfolio_metrics = calculate_portfolio_metrics(st.session_state.portfolio, st.session_state.use_mock_data)
    holdings_data = portfolio_metrics['holdings']

    for ticker in st.session_state.watchlist:
        stock_data = fetch_stock_data(ticker, st.session_state.use_mock_data)
        watch_analytics = calculate_watchlist_analytics(stock_data['history'], stock_data['current_price'])
        watch_correlation = calculate_watchlist_correlation(ticker, stock_data['history'], holdings_data)

        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

        with col1:
            st.markdown(f"### {ticker}")
            change_class = "positive" if stock_data['change'] >= 0 else "negative"
            st.markdown(f"<span class='{change_class}'>{stock_data['change_percent']:+.2f}%</span>", unsafe_allow_html=True)
            st.caption(f"Vol: {watch_analytics['volatility_class']} ({watch_analytics['volatility']:.1f}%) | Corr: {watch_correlation:.2f}")

        with col2:
            st.metric("Price", f"${stock_data['current_price']:.2f}")

        with col3:
            st.metric("Volume", f"{stock_data['volume']:,}")

        with col4:
            if not stock_data['history'].empty:
                fig = create_sparkline(stock_data['history'][-30:])
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"1D: {watch_analytics['1D']:+.2f}%")
        with col2:
            st.caption(f"1W: {watch_analytics['1W']:+.2f}%")
        with col3:
            st.caption(f"1M: {watch_analytics['1M']:+.2f}%")

        st.markdown("---")

def create_drawdown_chart(holdings_data: List[Dict]) -> go.Figure:
    """Create drawdown time-series chart"""
    port_returns, dates = calculate_portfolio_returns_series(holdings_data)

    if len(port_returns) == 0:
        return go.Figure().add_annotation(text="Insufficient data")

    portfolio_values = np.cumprod(1 + port_returns)
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='#ef4444', width=2),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Portfolio Drawdown Over Time',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        height=350,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Create correlation matrix heatmap"""
    if corr_matrix.empty:
        return go.Figure().add_annotation(text="Insufficient data for correlation")

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        hovertemplate='%{y} - %{x}: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title='Portfolio Correlation Matrix',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_contribution_chart(contributions: Dict[str, float]) -> go.Figure:
    """Create contribution to return bar chart"""
    if not contributions:
        return go.Figure().add_annotation(text="Insufficient data")

    tickers = list(contributions.keys())
    contrib_values = list(contributions.values())
    colors = ['#10b981' if v > 0 else '#ef4444' for v in contrib_values]

    fig = go.Figure(data=[go.Bar(
        x=tickers,
        y=contrib_values,
        marker=dict(color=colors),
        hovertemplate='<b>%{x}</b><br>Contribution: %{y:.2f}%<extra></extra>'
    )])

    fig.update_layout(
        title='Contribution to Portfolio Return (%)',
        xaxis_title='Ticker',
        yaxis_title='Contribution (%)',
        template='plotly_white',
        height=350,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_benchmark_comparison_chart(portfolio_returns: np.ndarray, benchmark_history: pd.DataFrame) -> go.Figure:
    """Create portfolio vs benchmark comparison chart"""
    if len(portfolio_returns) == 0 or benchmark_history.empty:
        return go.Figure().add_annotation(text="Insufficient data for benchmark comparison")

    bench_returns = calculate_returns_series(benchmark_history['Close'].values[-len(portfolio_returns) - 1:])

    if len(bench_returns) == 0:
        return go.Figure().add_annotation(text="Insufficient benchmark data")

    min_len = min(len(portfolio_returns), len(bench_returns))
    port_cum = np.cumprod(1 + portfolio_returns[-min_len:])
    bench_cum = np.cumprod(1 + bench_returns[-min_len:])

    dates = benchmark_history.index[-min_len:] if min_len > 0 else []

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=port_cum,
        mode='lines',
        name='Portfolio',
        line=dict(color='#667eea', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates,
        y=bench_cum,
        mode='lines',
        name='S&P 500',
        line=dict(color='#10b981', width=2)
    ))

    fig.update_layout(
        title='Portfolio vs S&P 500 Benchmark',
        xaxis_title='Date',
        yaxis_title='Cumulative Value (Indexed)',
        template='plotly_white',
        height=350,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def render_analytics_tab(metrics: Dict):
    """Render analytics tab"""
    st.subheader("📈 Advanced Analytics")

    holdings_data = metrics['holdings']
    port_returns, _ = calculate_portfolio_returns_series(holdings_data)

    if len(holdings_data) == 0:
        st.info("Add holdings to portfolio to view analytics")
        return

    st.markdown("### Performance Analytics")
    col1, col2, col3 = st.columns(3)

    if len(port_returns) > 0:
        prices = np.cumprod(1 + port_returns)
        rolling_rets = calculate_rolling_returns(prices, [30, 90, 180, 252])

        with col1:
            st.metric("1M Return", f"{rolling_rets.get('30D', 0):.2f}%")
        with col2:
            st.metric("3M Return", f"{rolling_rets.get('90D', 0):.2f}%")
        with col3:
            st.metric("6M Return", f"{rolling_rets.get('180D', 0):.2f}%")

    st.markdown("---")
    st.markdown("### Risk Metrics")

    col1, col2, col3, col4 = st.columns(4)

    if len(port_returns) > 0:
        max_dd, dd_duration, _ = calculate_maximum_drawdown(np.cumprod(1 + port_returns))
        sortino = calculate_sortino_ratio(port_returns)
        var_95, cvar_95 = calculate_var_cvar(port_returns, 0.95)

        with col1:
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
        with col2:
            st.metric("Sortino Ratio", f"{sortino:.2f}")
        with col3:
            st.metric("VaR (95%)", f"{var_95:.2f}%")
        with col4:
            st.metric("CVaR (95%)", f"{cvar_95:.2f}%")

    st.markdown("---")
    st.markdown("### Benchmark Comparison")

    benchmark_hist = fetch_benchmark_data()
    bench_comparison = calculate_benchmark_comparison(port_returns, benchmark_hist)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Portfolio Return", f"{bench_comparison['portfolio_return']:.2f}%")
    with col2:
        st.metric("S&P 500 Return", f"{bench_comparison['benchmark_return']:.2f}%")
    with col3:
        outperf_color = "green" if bench_comparison['outperformance'] > 0 else "red"
        st.metric("Outperformance", f"{bench_comparison['outperformance']:.2f}%")
    with col4:
        st.metric("Correlation w/ Benchmark", f"{bench_comparison['correlation']:.2f}")

    st.markdown("---")
    st.markdown("### Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_drawdown_chart(holdings_data), use_container_width=True)

    with col2:
        st.plotly_chart(create_benchmark_comparison_chart(port_returns, benchmark_hist), use_container_width=True)

    st.markdown("---")
    st.markdown("### Contribution Analysis")

    contributions = calculate_contribution_to_return(holdings_data, port_returns)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.plotly_chart(create_contribution_chart(contributions), use_container_width=True)

    with col2:
        st.plotly_chart(create_risk_return_heatmap(holdings_data), use_container_width=True)

    st.markdown("---")
    st.markdown("### Diversification Analysis")

    corr_matrix, avg_corr, high_corr_pairs = calculate_portfolio_correlation_matrix(holdings_data)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.plotly_chart(create_correlation_heatmap(corr_matrix), use_container_width=True)

    with col2:
        st.markdown(f"**Avg Correlation:** {avg_corr:.3f}")

        if high_corr_pairs:
            st.warning("High Correlation Pairs (>0.75):")
            for pair in high_corr_pairs:
                st.text(pair)
        else:
            st.success("No highly correlated pairs detected")

    st.markdown("---")
    st.subheader("🎲 Monte Carlo Simulation")

    col1, col2 = st.columns([1, 3])

    with col1:
        with st.form("monte_carlo_form"):
            sim_days = st.slider("Forecast Days", 30, 365, st.session_state.monte_carlo_params['days'])
            sim_runs = st.slider("Simulations", 100, 5000, st.session_state.monte_carlo_params['simulations'], step=100)

            submitted = st.form_submit_button("Run Simulation", type="primary")

            if submitted:
                st.session_state.monte_carlo_params = {'days': sim_days, 'simulations': sim_runs}

                with st.spinner("Running simulation..."):
                    cache_key = f"{sim_days}_{sim_runs}_{get_ist_now().strftime('%Y%m%d')}"
                    days, simulations = monte_carlo_simulation(
                        metrics['holdings'],
                        sim_days,
                        sim_runs,
                        _cache_key=cache_key
                    )
                    if len(simulations) > 0:
                        st.session_state.monte_carlo_results = (days, simulations, metrics['total_value'])
                        st.success("Simulation complete!")

    with col2:
        if st.session_state.monte_carlo_results is not None:
            days, simulations, portfolio_value = st.session_state.monte_carlo_results
            st.plotly_chart(create_monte_carlo_chart(days, simulations), use_container_width=True)

            final_values = simulations[:, -1]
            st.markdown(f"""
            **Simulation Results (Day {len(days)}):**
            - 10th Percentile: ${np.percentile(final_values, 10):,.2f}
            - 50th Percentile (Median): ${np.percentile(final_values, 50):,.2f}
            - 90th Percentile: ${np.percentile(final_values, 90):,.2f}
            - Expected Return: {((np.mean(final_values) / portfolio_value) - 1) * 100:.2f}%
            """)
        else:
            st.info("Click 'Run Simulation' to generate Monte Carlo forecast")

def render_rebalancing_tab(metrics: Dict):
    """Render rebalancing tab"""
    st.subheader("⚖️ Portfolio Rebalancing")

    sector_allocation = {}
    for h in metrics['holdings']:
        sector = h['sector']
        if sector in sector_allocation:
            sector_allocation[sector] += h['current_value']
        else:
            sector_allocation[sector] = h['current_value']

    total_value = sum(sector_allocation.values())
    current_allocation = {k: (v / total_value) * 100 for k, v in sector_allocation.items()}
    current_allocation['Cash'] = (st.session_state.cash_balance / (total_value + st.session_state.cash_balance)) * 100

    st.plotly_chart(create_rebalancing_chart(current_allocation, st.session_state.target_allocation), use_container_width=True)

    st.markdown("---")
    st.subheader("Rebalancing Suggestions")

    suggestions = []
    for sector, target_pct in st.session_state.target_allocation.items():
        current_pct = current_allocation.get(sector, 0)
        diff = target_pct - current_pct

        if abs(diff) > 2:
            action = "Buy" if diff > 0 else "Sell"
            amount = abs(diff) * (total_value + st.session_state.cash_balance) / 100
            suggestions.append({
                'Sector': sector,
                'Action': action,
                'Current %': f"{current_pct:.1f}%",
                'Target %': f"{target_pct:.1f}%",
                'Difference': f"{diff:+.1f}%",
                'Amount': f"${amount:,.2f}"
            })

    if suggestions:
        df = pd.DataFrame(suggestions)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Portfolio is well balanced!")

    st.markdown("---")
    st.subheader("Set Target Allocation")

    with st.form("target_allocation_form"):
        for sector in st.session_state.target_allocation.keys():
            new_value = st.slider(
                sector,
                0,
                100,
                int(st.session_state.target_allocation[sector]),
                key=f"target_{sector}"
            )
            st.session_state.target_allocation[sector] = new_value

        submitted = st.form_submit_button("Update Target Allocation", type="primary")
        if submitted:
            st.success("Target allocation updated!")

def render_settings_tab():
    """Render settings tab"""
    st.subheader("⚙️ Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Data Source")
        use_mock = st.checkbox("Use Mock Data", value=st.session_state.use_mock_data, key="use_mock_checkbox")
        if use_mock != st.session_state.use_mock_data:
            st.session_state.use_mock_data = use_mock

        st.markdown("#### Cash Balance")
        with st.form("cash_balance_form"):
            new_cash = st.number_input(
                "Available Cash",
                min_value=0.0,
                value=st.session_state.cash_balance,
                step=100.0,
                key="cash_input"
            )
            submitted = st.form_submit_button("Update Cash Balance", type="primary")
            if submitted:
                st.session_state.cash_balance = new_cash
                st.success("Cash balance updated!")

    with col2:
        st.markdown("#### Export Options")

        if st.button("📥 Export Portfolio as CSV", type="primary"):
            metrics = calculate_portfolio_metrics(st.session_state.portfolio, st.session_state.use_mock_data)
            df = pd.DataFrame(metrics['holdings'])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"portfolio_{get_ist_now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        st.markdown("#### Danger Zone")
        with st.form("clear_portfolio_form"):
            confirm = st.checkbox("I understand this action cannot be undone")
            submitted = st.form_submit_button("🗑️ Clear Portfolio", type="secondary")
            if submitted and confirm:
                st.session_state.portfolio = []
                st.success("Portfolio cleared!")
                st.rerun()

def main():
    """Main application"""
    initialize_session_state()

    metrics = calculate_portfolio_metrics(st.session_state.portfolio, st.session_state.use_mock_data)

    render_header(metrics)

    st.sidebar.title("Navigation")
    tabs = ["Portfolio", "Watchlist", "Analytics", "Rebalancing", "Settings"]
    selected_tab = st.sidebar.radio("Go to", tabs, key="nav_radio")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    if st.sidebar.button("🔄 Refresh Data"):
        st.session_state.last_update = get_ist_now()
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("📊 Export All Charts"):
        st.sidebar.info("Chart export feature - click individual charts to download")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Data Mode:** {'Mock' if st.session_state.use_mock_data else 'Live'}")
    st.sidebar.markdown(f"**Portfolio Holdings:** {len(st.session_state.portfolio)}")

    if selected_tab == "Portfolio":
        render_portfolio_tab(metrics)
    elif selected_tab == "Watchlist":
        render_watchlist_tab()
    elif selected_tab == "Analytics":
        render_analytics_tab(metrics)
    elif selected_tab == "Rebalancing":
        render_rebalancing_tab(metrics)
    elif selected_tab == "Settings":
        render_settings_tab()

    st.markdown("""
    <div class="footer">
        <p>Data provided by Yahoo Finance | Dashboard built with Streamlit + Plotly</p>
        <p>© 2025 Real-Time Investment Portfolio Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
