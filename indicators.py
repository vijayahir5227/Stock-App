import pandas as pd

def add_indicators(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    delta     = df['Close'].diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    rolling_mean            = df['Close'].rolling(20).mean()
    rolling_std             = df['Close'].rolling(20).std()
    df['Bollinger_Upper']   = rolling_mean + (2 * rolling_std)
    df['Bollinger_Lower']   = rolling_mean - (2 * rolling_std)
    df['Bollinger_Mid']     = rolling_mean

    ema12            = df['Close'].ewm(span=12, adjust=False).mean()
    ema26            = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']       = ema12 - ema26
    df['MACD_Signal']= df['MACD'].ewm(span=9, adjust=False).mean()

    df = df.dropna()
    return df