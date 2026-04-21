import yfinance as yf
import pandas as pd

def load_data(symbol, period="2y"):
    try:
        df = yf.download(symbol, period=period,
                         auto_adjust=True, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.loc[:, ~df.columns.duplicated()]
        df = df[['Close']].dropna()

        if df.empty or len(df) < 50:
            return None

        return df

    except Exception as e:
        print(f"Failed to load {symbol}: {e}")
        return None