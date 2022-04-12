import pandas as pd


def add_diff_column(df: pd.DataFrame) -> pd.DataFrame:
    for col_name in df.columns.values:
        df[f"{col_name}_diff"] = df[col_name].diff()
    return df


def remove_unchanged_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assume that every row must either change together or stay the same together
    """
    first_col_name = df.columns.values[0]
    unchanged_filt = df[first_col_name].diff() == 0
    df = df.loc[~unchanged_filt]
    return df


def compensate_diff_with_cashflow(
    df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    acc_name_l: list[str],
) -> pd.DataFrame:
    for acc_name in acc_name_l:
        df[f"{acc_name}_diff"] = df[f"{acc_name}_diff"].add(
            cashflow_df[acc_name],
            fill_value=0,
        )
    return df


def add_acc_combined_cols(
    df: pd.DataFrame,
    acc_combined_name: str,
    acc_name_l: list[str],
) -> pd.DataFrame:
    # Combined col is sum of individual acc cols
    df[acc_combined_name] = df[acc_name_l].sum(axis=1)

    # Combined diff col is sum of individual acc diff cols
    diff_acc_column_name_l = [f"{acc_name}_diff" for acc_name in acc_name_l]
    df[f"{acc_combined_name}_diff"] = df[diff_acc_column_name_l].sum(axis=1)
    return df

def add_index_combined_cols(
    df: pd.DataFrame,
    index_combined_name: str,
    index_name_l: list[str],
) -> pd.DataFrame:
    # Combined col is mean of individual index cols
    df[index_combined_name] = df[index_name_l].mean(axis=1)

    # Combined diff col is mean of individual index diff cols
    diff_index_column_name_l = [f"{index_name}_diff" for index_name in index_name_l]
    df[f"{index_combined_name}_diff"] = df[diff_index_column_name_l].mean(axis=1)
    return df