import pandas as pd

def add_diff_column(df: pd.DataFrame) -> pd.DataFrame:
    for col_name in df.columns.values:
        df[f"{col_name}_diff"] = df[col_name].diff()
    return df