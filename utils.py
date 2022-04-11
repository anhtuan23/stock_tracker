import numpy as np
import pandas as pd
from pyxirr import xirr


def add_labels(ax, x, y, label_l=None, color=None):
    if label_l is None:
        label_l = [f"{y_i:.1f}" for y_i in y]
    for xi, yi, label in zip(x, y, label_l):
        max_y = max(y)
        y_pos_delta = max_y * 0.15
        y_pos_delta = y_pos_delta if yi > 0 else -y_pos_delta
        ax.text(xi, yi + y_pos_delta, label, ha="center", color=color)


def add_trend_line(ax, ticks, x, y):
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(ticks, p(x), ":", alpha=0.7)


def calc_cashflow_xirr(
    cashflow_df: pd.DataFrame,
    log_df: pd.DataFrame,
    anchor_date: str | None,
    date_idx: pd.Timestamp,
    col_name: str,
    user_name_combined_l: list[str],
) -> float:
    # remove cashflow after date_idx
    cashflow_df = cashflow_df.loc[cashflow_df.index <= date_idx]  # type: ignore

    # if anchor_date is provided, we have to determined the amount at anchor date,
    # and only use cashflow after that anchor date
    if anchor_date is not None:
        cashflow_df = cashflow_df.loc[cashflow_df.index >= anchor_date]  # type: ignore
        # earlier date is the last date before anchor date
        earlier_df = log_df.loc[log_df.index < anchor_date]  # type: ignore
        if not earlier_df.empty:
            anchor_date_idx = earlier_df.index[-1]

            # Note: earliest amount should be negative since it's viewed as an investment
            earliest_amount_dict = {
                name: -log_df.loc[anchor_date_idx, name]  # type: ignore
                for name in user_name_combined_l
            }
            earliest_amount_df = pd.DataFrame(
                earliest_amount_dict, index=[anchor_date_idx]
            )

            cashflow_df = pd.concat([earliest_amount_df, cashflow_df])

    latest_amount_dict = {
        name: log_df.loc[date_idx, name] for name in user_name_combined_l  # type: ignore
    }
    latest_amount_df = pd.DataFrame(latest_amount_dict, index=[date_idx])

    xirr_cf_df = pd.concat([cashflow_df, latest_amount_df])

    amounts = xirr_cf_df[col_name]
    if all(amounts == 0):  # type: ignore
        return 0

    xirr_val = xirr(xirr_cf_df.index, amounts)  # type: ignore
    return 0 if xirr_val is None else xirr_val


def calc_index_xirr(
    log_dataframe: pd.DataFrame,
    anchor_date: str | None,
    date_idx: pd.Timestamp,
    index_name: str,
) -> float:
    if anchor_date is not None:
        earlier_df = log_dataframe.loc[log_dataframe.index < anchor_date]  # type: ignore

        # anchor_date_idx should be the last date before anchor date
        anchor_date_idx = anchor_date if earlier_df.empty else earlier_df.index[-1]
        log_dataframe = log_dataframe.loc[log_dataframe.index >= anchor_date_idx]  # type: ignore
    first_idx = log_dataframe.index[0]

    return xirr(
        [first_idx, date_idx],  # type: ignore
        [
            -log_dataframe.loc[first_idx, index_name],  # type: ignore
            log_dataframe.loc[date_idx, index_name],  # type: ignore
        ],
    )


def read_acc_cashflow(
    acc_name: str,
    user_name_list: list[str],
) -> pd.DataFrame:
    cf_df_list: list[pd.DataFrame] = []
    for user_name in user_name_list:
        cf_df = pd.read_csv(
            f"./stock_data - {user_name}_cashflow.csv",
            index_col="date",
            parse_dates=True,
        )
        cf_df_list.append(cf_df)

    acc_cf_df: pd.DataFrame = pd.concat(cf_df_list, axis=1)
    acc_cf_df[acc_name] = acc_cf_df.sum(axis=1)

    return acc_cf_df
