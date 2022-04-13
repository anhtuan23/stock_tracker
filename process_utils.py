import pandas as pd
import numpy as np
import utils


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


def add_diff_percent(
    df: pd.DataFrame,
    acc_name_combined_l: list[str],
    index_name_combined_l: list[str],
) -> pd.DataFrame:
    """Add diff percent and auxiliary diff percent"""
    for name in acc_name_combined_l + index_name_combined_l:

        df[f"{name}_diff_p"] = df[f"{name}_diff"] / df[name].shift()

        # Replace inf values with nan in diff_p (otherwise, growth would be infinite)
        df[f"{name}_diff_p"] = df[f"{name}_diff_p"].replace([np.inf, -np.inf], np.nan)

        df[f"{name}_aux_diff_p"] = df[f"{name}_diff_p"] + 1

    return df


def filter_latest_x_rows(df: pd.DataFrame, row_num: int) -> pd.DataFrame:
    return df.iloc[-row_num:]  # type: ignore


def add_period_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["Y"] = df.index.to_period("Y")  # type: ignore
    df["Q"] = df.index.to_period("Q")  # type: ignore
    df["M"] = df.index.to_period("M")  # type: ignore
    df["W"] = df.index.to_period("W")  # type: ignore
    return df


def get_period_df(
    period_symbol: str,
    df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    acc_name_combined_l: list[str],
    index_name_combined_l: list[str],
) -> pd.DataFrame:
    period_l = df[period_symbol].unique()

    data = []
    for period in period_l:
        period_data_dict = {"period": period}
        period_filt = df[period_symbol] == period
        period_df = df.loc[period_filt]

        for name in acc_name_combined_l + index_name_combined_l:
            period_growth = period_df[f"{name}_aux_diff_p"].product()  # type: ignore
            period_data_dict[f"{name}_growth"] = (period_growth - 1) * 100  # type: ignore

        for acc_name in acc_name_combined_l:
            period_xirr = utils.calc_cashflow_xirr(
                cashflow_df,
                df,
                anchor_date=period_df.index[0],  # type: ignore
                date_idx=period_df.index[-1],  # type: ignore
                col_name=acc_name,
                user_name_combined_l=acc_name_combined_l,
            )  # type: ignore
            period_data_dict[f"{acc_name}_xirr"] = period_xirr * 100  # type: ignore

            period_data_dict[f"{acc_name}_income"] = period_df[f"{acc_name}_diff"].sum()  # type: ignore

        for index_name in index_name_combined_l:
            period_xirr = utils.calc_index_xirr(
                df,
                anchor_date=period_df.index[0],  # type: ignore
                date_idx=period_df.index[-1],  # type: ignore
                index_name=index_name,
            )  # type: ignore
            period_data_dict[f"{index_name}_xirr"] = period_xirr * 100  # type: ignore

        data.append(period_data_dict)

    period_df = pd.DataFrame(data)
    period_df = period_df.set_index("period")
    period_df.index = period_df.index.to_series().astype(str)
    return period_df
