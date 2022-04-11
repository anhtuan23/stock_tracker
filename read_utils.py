import pandas as pd

def read_log() -> pd.DataFrame:
    log_df = pd.read_csv("./stock_data - log.csv", index_col="date", parse_dates=True)
    log_df = log_df.sort_index()
    assert log_df.index.inferred_type == "datetime64"
    return log_df