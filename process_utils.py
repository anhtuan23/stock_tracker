import pandas as pd
import numpy as np

# TODO: remove const
import utils, read_utils, const


def _add_diff_column(df: pd.DataFrame) -> pd.DataFrame:
    for col_name in df.columns.values:
        df[f"{col_name}_diff"] = df[col_name].diff()
    return df


def _remove_unchanged_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assume that every row must either change together or stay the same together
    """
    first_col_name = df.columns.values[0]
    unchanged_filt = df[first_col_name].diff() == 0
    df = df.loc[~unchanged_filt]
    return df


def _compensate_diff_with_cashflow(
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


def _add_acc_combined_cols(
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


def _add_index_combined_cols(
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


def _add_diff_percent(
    df: pd.DataFrame,
    all_acc_name_l: list[str],
    index_name_combined_l: list[str],
) -> pd.DataFrame:
    """Add diff percent and auxiliary diff percent"""
    for name in all_acc_name_l + index_name_combined_l:

        df[f"{name}_diff_p"] = df[f"{name}_diff"] / (df[name] - df[f"{name}_diff"])

        # Replace inf values with nan in diff_p (otherwise, growth would be infinite)
        df[f"{name}_diff_p"] = df[f"{name}_diff_p"].replace([np.inf, -np.inf], np.nan)

        df[f"{name}_aux_diff_p"] = df[f"{name}_diff_p"] + 1

    return df


def prepare_log_df_cf_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Read log table
    log_df = read_utils.read_log()

    # Read Cashflow
    cf_df = read_utils.read_cashflow(const.ACC_USER_DICT, const.ACC_COMBINED_NAME)

    # Add diff columns
    log_df = _add_diff_column(log_df)

    # Remove unchanged rows
    log_df = _remove_unchanged_rows(log_df)

    # Compensate diff with cashflow
    log_df = _compensate_diff_with_cashflow(
        df=log_df,
        cashflow_df=cf_df,
        acc_name_l=const.ACC_NAME_L,
    )

    # Add acc combined cols
    log_df = _add_acc_combined_cols(
        log_df,
        acc_combined_name=const.ACC_COMBINED_NAME,
        acc_name_l=const.ACC_NAME_L,
    )

    # Add index combined cols
    log_df = _add_index_combined_cols(
        log_df,
        index_combined_name=const.INDEX_COMBINED_NAME,
        index_name_l=const.INDEX_NAME_L,
    )

    # Calculate diff percent & auxiliary diff percent
    log_df = _add_diff_percent(log_df, const.ALL_ACC_NAME_L, const.ALL_INDEX_NAME_L)

    return log_df, cf_df


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
    all_acc_name_l: list[str],
    index_name_combined_l: list[str],
) -> pd.DataFrame:
    period_l = df[period_symbol].unique()

    data = []
    for period in period_l:
        period_data_dict = {"period": period}
        period_filt = df[period_symbol] == period
        period_df = df.loc[period_filt]

        for name in all_acc_name_l + index_name_combined_l:
            period_growth = period_df[f"{name}_aux_diff_p"].product()  # type: ignore
            period_data_dict[f"{name}_growth"] = (period_growth - 1) * 100  # type: ignore

        for acc_name in all_acc_name_l:
            period_xirr = utils.calc_cashflow_xirr(
                cashflow_df,
                df,
                anchor_date=period_df.index[0],  # type: ignore
                date_idx=period_df.index[-1],  # type: ignore
                col_name=acc_name,
                user_name_combined_l=all_acc_name_l,
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


### User Analysis


def get_user_df(
    log_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    acc_name: str,
    user_name_l: list[str],
    index_name: str,
) -> pd.DataFrame:
    # Get data from log_df
    acc_df = log_df[[index_name, f"{index_name}_diff", acc_name, f"{acc_name}_diff"]]

    # Get data from cashflow dataframe
    acc_cf_df = cf_df[user_name_l]
    acc_cf_df[f"{acc_name}_cf"] = acc_cf_df.sum(axis=1)
    acc_cf_df = acc_cf_df.rename(
        columns={user_name: f"{user_name}_cf" for user_name in user_name_l}
    )

    user_df = pd.concat([acc_df, acc_cf_df], axis=1)
    user_df.fillna(0, inplace=True)

    user_df["day_start"] = user_df[acc_name] - user_df[f"{acc_name}_diff"]

    # Delete rows with day_start == 0 (the first row in this case)
    user_df = user_df[user_df["day_start"] != 0]

    # Calculating share of each user
    yesterday_user_nav_dict = {user_name: 0.0 for user_name in user_name_l}
    for date in user_df.index:
        day_start: float = user_df.loc[date, "day_start"]  # type:ignore

        for user_name in user_name_l:
            # Get day start nav using yesterday nav and today cashflow
            user_day_start: float = (
                # Investment is saved as negative number in cashflow
                yesterday_user_nav_dict[user_name]
                - user_df.loc[date, f"{user_name}_cf"]  # type:ignore
            )

            user_df.loc[date, f"{user_name}_day_start"] = user_day_start

            user_share = user_day_start / day_start
            user_df.loc[date, f"{user_name}_share"] = user_share

            user_diff: float = (
                user_df.loc[date, f"{acc_name}_diff"] * user_share  # type:ignore
            )
            user_df.loc[date, f"{user_name}_diff"] = user_diff

            user_day_end_nav = user_day_start + user_diff
            user_df.loc[date, f"{user_name}"] = user_day_end_nav
            yesterday_user_nav_dict[user_name] = user_day_end_nav

    user_df = _add_diff_percent(
        user_df,
        all_acc_name_l=[acc_name],
        index_name_combined_l=[index_name],
    )

    return user_df


def get_overall_growth_xirr_df(
    log_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    anchor_date: str,
    acc_name_l: list[str],
    index_name_l: list[str],
) -> pd.DataFrame:
    recent_log_df: pd.DataFrame = log_df[log_df.index >= anchor_date]  # type: ignore
    growth_xirr_df = pd.DataFrame(index=recent_log_df.index)
    for name in acc_name_l + index_name_l:
        # calculate growth using cumulative product since anchor date

        growth_xirr_df[f"{name}_growth"] = (
            recent_log_df[f"{name}_aux_diff_p"].cumprod() * 100
        )

    growth_xirr_df = growth_xirr_df.fillna(100)

    for name in acc_name_l:
        growth_xirr_df[f"{name}_xirr"] = (
            recent_log_df.index.to_series().apply(
                lambda date_idx: utils.calc_cashflow_xirr(
                    cf_df,
                    log_df,
                    anchor_date,
                    date_idx,
                    name,
                    acc_name_l,
                )
            )
            * 100
        )
        # The first few xirr are too crazy to be included
        growth_xirr_df.iloc[:5, growth_xirr_df.columns.get_loc(f"{name}_xirr")] = 0

    for name in index_name_l:
        growth_xirr_df[f"{name}_xirr"] = (
            recent_log_df.index.to_series().apply(
                lambda date_idx: utils.calc_index_xirr(
                    log_df,
                    anchor_date,
                    date_idx,
                    name,
                )
            )
            * 100
        )
        # The first few xirr are too crazy to be included
        growth_xirr_df.iloc[:5, growth_xirr_df.columns.get_loc(f"{name}_xirr")] = 0

    return growth_xirr_df
