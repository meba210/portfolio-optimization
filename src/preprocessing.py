import pandas as pd
import numpy as np


def clean_data(df):
    df = df.copy()
    df = df.dropna()
    df["Return"] = df["Adj Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=20).std()
    return df.dropna()
