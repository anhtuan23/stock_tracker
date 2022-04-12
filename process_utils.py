import pandas as pd

def add_diff_column(df: pd.DataFrame) -> pd.DataFrame:
    for col_name in df.columns.values:
        df[f"{col_name}_diff"] = df[col_name].diff()
    return df

def remove_unchanged_rows(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Assume that every row must either change together or stay the same together
    '''
    first_col_name = df.columns.values[0]
    unchanged_filt = df[first_col_name].diff() == 0
    df = df.loc[~unchanged_filt]
    return df