import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting


def optimize_portfolio(price_df):
    """
    Optimize portfolio using PyPortfolioOpt.

    Parameters:
    -----------
    price_df : pd.DataFrame
        DataFrame with Adj Close prices for all assets (columns = tickers).

    Returns:
    --------
    weights : dict
        Optimal portfolio weights.
    performance : tuple
        Expected return, volatility, Sharpe ratio
    """
    # 1. Compute expected returns and sample covariance matrix
    mu = expected_returns.mean_historical_return(price_df)
    S = risk_models.sample_cov(price_df)

    # 2. Create Efficient Frontier object
    ef = EfficientFrontier(mu, S)

    # 3. Max Sharpe Portfolio
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # 4. Portfolio performance
    performance = ef.portfolio_performance(verbose=True)

    return cleaned_weights, performance, ef


def generate_efficient_frontier_plot(ef, title="Efficient Frontier"):
    """
    Plot the Efficient Frontier.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
    ax.set_title(title)
    return fig, ax
