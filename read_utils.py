import pandas as pd

def read_log() -> pd.DataFrame:
    log_df = pd.read_csv("./stock_data - log.csv", index_col="date", parse_dates=True)
    log_df = log_df.sort_index()
    assert log_df.index.inferred_type == "datetime64"
    return log_df

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