import pandas as pd


def read_log() -> pd.DataFrame:
    log_df = pd.read_csv("./stock_data - log.csv", index_col="date", parse_dates=True)
    log_df = log_df.sort_index()
    assert log_df.index.inferred_type == "datetime64"
    return log_df


def _read_acc_cashflow(
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


def read_cashflow(
    acc_user_dict: dict[str, list[str]],
    acc_combined_name: str | None,
) -> pd.DataFrame:
    acc_cf_df_list: list[pd.DataFrame] = []
    for acc_name, user_name_l_ in acc_user_dict.items():
        acc_cf_df = _read_acc_cashflow(acc_name, user_name_l_)
        acc_cf_df_list.append(acc_cf_df)

    cf_df = pd.concat(acc_cf_df_list, axis=1)
    assert cf_df.index.inferred_type == "datetime64"
    cf_df.fillna(0, inplace=True)

    if acc_combined_name is not None:
        acc_only_cf_df = cf_df[list(acc_user_dict.keys())]
        cf_df[acc_combined_name] = acc_only_cf_df.sum(axis=1)

    return cf_df
