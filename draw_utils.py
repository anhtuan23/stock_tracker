import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils
import process_utils


def plot_indices_over_time(
    time_series: list,
    index_values_dict: dict[str, list[float]],
) -> None:
    """
    index_values_dict: dict of {index_name: index_value_l}
    """
    plt.figure(figsize=(30, 5))
    for index_name in index_values_dict:
        plt.plot_date(
            time_series,
            index_values_dict[index_name],
            label=index_name,
            linestyle="-",
            marker=None,
        )

    plt.title("Index over time")

    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_period_growth_xirr(
    growth_xirr_df: pd.DataFrame,
    period_name: str,
    acc_name_l: list[str],
    index_name_l: list[str],
    acc_combined_name: str,
    index_combined_name: str,
    all_acc_name_l: list[str],
):

    fig, (ax1, ax2, ax3, ax4, income_ax) = plt.subplots(
        nrows=5,
        ncols=1,
        sharex=True,
        figsize=(28, 16),
    )  # type: ignore

    # ****** GROWTH ******
    def _plot(
        net_ax: plt.Axes,
        diff_ax: plt.Axes,
        type: str,
        income_ax: plt.Axes | None = None,
    ):
        """
        Parameters
        type: is either 'growth' or 'xirr'
        """
        # * Draw bar chart

        x_indices = np.arange(len(growth_xirr_df.index))

        bar_name_l = acc_name_l + index_name_l
        bar_count = len(bar_name_l)
        total_width = 0.6
        bar_width = total_width / bar_count

        for i, name in enumerate(bar_name_l):
            position = x_indices + i * bar_width
            net_ax.bar(
                position,
                growth_xirr_df[f"{name}_{type}"],
                label=name,
                width=bar_width,
            )

        # move back half bar width to left most edge and then move to the middle of the bar group
        ticks = x_indices - (0.5 * bar_width) + (bar_count / 2 * bar_width)

        # * Draw line chart

        for name, color in zip(
            [acc_combined_name, index_combined_name], ["orange", "orchid"]
        ):
            net_ax.plot(
                ticks,
                growth_xirr_df[f"{name}_{type}"],
                label=name,
                linestyle="--",
                c=color,
            )

        net_ax.fill_between(
            ticks,
            growth_xirr_df[f"{acc_combined_name}_{type}"],
            growth_xirr_df[f"{index_combined_name}_{type}"],
            where=(
                growth_xirr_df[f"{acc_combined_name}_{type}"]
                >= growth_xirr_df[f"{index_combined_name}_{type}"]  # type: ignore
            ),
            color="green",
            interpolate=True,
            alpha=0.25,
        )

        net_ax.fill_between(
            ticks,
            growth_xirr_df[f"{acc_combined_name}_{type}"],
            growth_xirr_df[f"{index_combined_name}_{type}"],
            where=(
                growth_xirr_df[f"{acc_combined_name}_{type}"] < growth_xirr_df[f"{index_combined_name}_{type}"]  # type: ignore
            ),
            color="red",
            interpolate=True,
            alpha=0.25,
        )

        # * Decorations
        y = growth_xirr_df[f"{acc_combined_name}_{type}"]  # type: ignore
        utils.add_labels(
            ax=net_ax,
            x=ticks,
            y=y,
            color="orange",
        )

        net_ax.set_title(f"{period_name} {type}")
        net_ax.set_ylabel(f"{type} %")
        net_ax.legend(loc="upper left")

        net_ax.grid(True)

        # *** Diff ***
        diff_series = (
            growth_xirr_df[f"{acc_combined_name}_{type}"] - growth_xirr_df[f"{index_combined_name}_{type}"]  # type: ignore
        )
        diff_ax.bar(
            ticks,
            diff_series,
            width=bar_width,
            label="Diff (user - index)",
            color="dodgerblue",
        )
        utils.add_labels(
            ax=diff_ax,
            x=ticks,
            y=diff_series,
            color="dodgerblue",
        )
        y = growth_xirr_df[f"{index_combined_name}_{type}"]
        utils.add_labels(
            ax=diff_ax,
            x=ticks,
            y=y,
            color="orchid",
        )

        utils.add_trend_line(diff_ax, ticks, ticks, diff_series)

        # Growth line
        for name, color in zip(
            [acc_combined_name, index_combined_name], ["orange", "orchid"]
        ):
            diff_ax.plot(
                ticks,
                growth_xirr_df[f"{name}_{type}"],
                label=name,
                marker="o",
                linestyle="-",
                color=color,
            )

        win_num = diff_series[diff_series > 0].count()  # type: ignore
        win_rate = win_num / len(diff_series) * 100

        diff_ax.set_title(f"Diff in {type} - win rate: {win_rate:.2f}%")
        diff_ax.set_ylabel(f"{type} %")
        diff_ax.legend(loc="upper left")
        diff_ax.grid(True)

        # Income
        if income_ax is not None:
            for name, color in zip(
                all_acc_name_l,
                ["dodgerblue", "orchid", "orange"],
            ):
                income_l = growth_xirr_df[f"{name}_income"]  # type: ignore
                income_ax.plot(
                    ticks,
                    income_l,
                    label=name,
                    color=color,
                )
                income_label_l = [f"{income:,.0f}" for income in income_l]
                utils.add_labels(
                    ax=income_ax,
                    x=ticks,
                    y=income_l,
                    label_l=income_label_l,
                    color=color,
                )
            income_ax.set_title(f"{period_name} income")
            income_ax.grid(True)
            income_ax.legend(loc="upper left")
            income_ax.set_xticks(
                ticks=ticks,
                labels=growth_xirr_df.index,
                rotation=30,
            )

    _plot(ax1, ax2, "growth", income_ax=income_ax)
    _plot(ax3, ax4, "xirr")

    plt.show()


def plot_nav_stackplot(x: list, y: list, labels: list[str]):
    fig, ax = plt.subplots(figsize=(26, 8))
    ax.stackplot(x, y, labels=labels)

    # set labels to absolute values
    # ax.set_yticklabels(abs(ax.get_yticks() / 1_000_000))

    ax.set_title("NAV over time")
    ax.legend(loc="upper left")
    plt.show()


def plot_nav_pie(nav_l: list, label_l: list[str]):
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="white")
    ax.pie(nav_l, labels=label_l, autopct="%1.1f%%")
    ax.set_title(f"Total Nav:{np.sum(nav_l):,}")
    plt.show()


def plot_profit_invest_pies(profit_invest_list: list[tuple[str, float, float]]):
    """
    profit_invest_list: list of tuple[name, profit, invest]
    """
    chart_num = len(profit_invest_list)
    width_per_pie = 6
    fig, ax_l = plt.subplots(figsize=(width_per_pie * chart_num, 5), ncols=chart_num, facecolor="white")  # type: ignore

    for (name, profit, invest), ax in zip(profit_invest_list, ax_l):  # type: ignore
        nav = profit + invest
        # If profit is positive, plot pie chart of profit & invest
        if profit > 0:
            profit_label = f"Profit:{profit:,.0f}"
            invest_label = f"Invest:{invest:,.0f}"
            ax.pie(
                [profit, invest],
                labels=[profit_label, invest_label],
                autopct="%1.1f%%",
                colors=["green", "orange"],
            )
            ax.set_title(f"{name} NAV:{nav:,.0f}")
        # Else, plot pie chart of NAV & lost
        else:
            loss = -profit
            loss_label = f"Loss:{loss:,.0f}"
            nav_label = f"NAV:{nav:,.0f}"
            ax.pie(
                [loss, nav],
                labels=[loss_label, nav_label],
                autopct="%1.1f%%",
                colors=["red", "orange"],
            )
            ax.set_title(f"{name} Invest:{invest:,.0f}")


def _plot_cum_growth(
    ax: plt.Axes,
    day_num: int,
    acc_name: str,
    cum_acc_growth: float,
    index_name: str,
    cum_index_growth: float,
) -> None:
    cum_diff_growth = cum_acc_growth - cum_index_growth
    x = [acc_name, index_name, "diff"]
    y = [cum_acc_growth, cum_index_growth, cum_diff_growth]
    ax.bar(
        x,
        y,
        width=0.5,
        alpha=0.8,
        color="dodgerblue",
    )
    ax.set_title(f"Cumulative growth for last {day_num} days")
    ax.grid(True)
    utils.add_labels(ax, x, y, color="dodgerblue")


def plot_recent_growth(
    log_df: pd.DataFrame,
    num_days: int,
    main_acc_name: str,
    main_index_name: str,
    secondary_acc_name_l: list[str],
    secondary_index_name_l: list[str],
):
    # Filter recent days
    recent_df = process_utils.filter_latest_x_rows(log_df, row_num=num_days)

    fig, (ax1, ax2, ax3) = plt.subplots(
        figsize=(28, 12),
        ncols=3,
        sharey=True,
        gridspec_kw={"width_ratios": [4, 1, 1]},
    )  # type: ignore

    # * Last x days line plot
    ## Solid main lines
    for solid_name, color in zip(
        [main_acc_name, main_index_name],
        ["lightseagreen", "orange"],
    ):
        x = recent_df.index
        y = recent_df[f"{solid_name}_diff_p"] * 100
        ax1.plot_date(
            x,
            y,
            fmt="-",
            label=solid_name,
            color=color,
        )
        utils.add_labels(ax1, x, y, color=color)

    recent_diff_series = (
        recent_df[f"{main_acc_name}_diff_p"] - recent_df[f"{main_index_name}_diff_p"]
    ) * 100

    ax1.bar(
        recent_df.index,
        recent_diff_series,
        width=0.5,
        alpha=0.8,
        color="dodgerblue",
    )
    utils.add_labels(ax1, recent_df.index, recent_diff_series, color="dodgerblue")

    # Trendline
    first_date = recent_df.index[0]
    x = [(date - first_date).days for date in recent_df.index]
    utils.add_trend_line(ax1, ticks=recent_df.index, x=x, y=recent_diff_series)

    for single_name in secondary_acc_name_l:
        ax1.plot_date(
            recent_df.index,
            recent_df[f"{single_name}_diff_p"] * 100,
            fmt="-.",
            alpha=0.7,
            label=single_name,
        )

    for single_name in secondary_index_name_l:
        ax1.plot_date(
            recent_df.index,
            recent_df[f"{single_name}_diff_p"] * 100,
            fmt=":",
            alpha=0.7,
            label=single_name,
        )

    ax1.set_title(f"Growth for last {num_days} days")
    ax1.set_xticks(ticks=recent_df.index)
    ax1.legend()
    ax1.grid(True)

    # # * Cumulative growth bar plot
    for ax in [ax2, ax3]:
        recent_df = process_utils.filter_latest_x_rows(recent_df, row_num=num_days)

        cum_acc_growth = (recent_df[f"{main_acc_name}_aux_diff_p"].product() - 1) * 100  # type: ignore
        cum_index_growth = (recent_df[f"{main_index_name}_aux_diff_p"].product() - 1) * 100  # type: ignore
        _plot_cum_growth(
            ax=ax,
            day_num=num_days,
            acc_name=main_acc_name,
            cum_acc_growth=cum_acc_growth,
            index_name=main_index_name,
            cum_index_growth=cum_index_growth,
        )

        num_days = num_days // 2

    plt.show()


def plot_recent_xirr(
    log_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    num_days: int,
    anchor_date: str,
    main_acc_name: str,
    main_index_name: str,
    secondary_acc_name_l: list[str],
    secondary_index_name_l: list[str],
):
    """
    anchor_date: str, e.g. "2020-01-31"
    """
    # Filter recent days
    recent_df = process_utils.filter_latest_x_rows(log_df, row_num=num_days)

    all_acc_name_l = [main_acc_name] + secondary_acc_name_l
    all_index_name_l = [main_index_name] + secondary_index_name_l

    xirr_data = {name: [] for name in all_acc_name_l + all_index_name_l}
    for idx in recent_df.index:
        for name in all_acc_name_l:
            xirr_data[name].append(
                utils.calc_cashflow_xirr(
                    cashflow_df=cf_df,
                    log_df=log_df,
                    anchor_date=anchor_date,
                    date_idx=idx,
                    col_name=name,
                    user_name_combined_l=all_acc_name_l,
                )
            )
        for idx_name in all_index_name_l:
            xirr_data[idx_name].append(
                utils.calc_index_xirr(log_df, anchor_date, idx, idx_name)
            )
    xirr_df = pd.DataFrame(xirr_data, index=recent_df.index)

    fig, ax1 = plt.subplots(figsize=(18, 6))  # type: ignore

    # * Last x days line plot

    for combined_name in [main_acc_name, main_index_name]:
        ax1.plot_date(
            xirr_df.index,
            xirr_df[combined_name] * 100,  # type: ignore
            fmt="-",
            label=combined_name,
        )
        utils.add_labels(ax1, xirr_df.index, xirr_df[combined_name] * 100)  # type: ignore

        # Trendline
        first_date = xirr_df.index[0]
        x = [(date - first_date).days for date in xirr_df.index]
        utils.add_trend_line(ax1, ticks=xirr_df.index, x=x, y=xirr_df[combined_name] * 100)  # type: ignore

    recent_diff = (xirr_df[main_acc_name] - xirr_df[main_index_name]) * 100  # type: ignore

    ax1.bar(
        xirr_df.index,
        recent_diff,
        width=0.5,
        alpha=0.8,
    )
    utils.add_labels(ax1, xirr_df.index, recent_diff)  # type: ignore

    # Trendline
    first_date = xirr_df.index[0]
    x = [(date - first_date).days for date in xirr_df.index]
    utils.add_trend_line(ax1, ticks=xirr_df.index, x=x, y=recent_diff)  # type: ignore

    for single_name in secondary_acc_name_l:
        ax1.plot_date(
            xirr_df.index,
            xirr_df[single_name] * 100,  # type: ignore
            fmt="-.",
            alpha=0.7,
            label=single_name,
        )

    for single_name in secondary_index_name_l:
        ax1.plot_date(
            xirr_df.index,
            xirr_df[single_name] * 100,  # type: ignore
            fmt=":",
            alpha=0.7,
            label=single_name,
        )

    ax1.set_title(f"XIRR from {anchor_date} for last 10 days")
    ax1.set_xticks(ticks=xirr_df.index)
    ax1.legend(loc="center left")
    ax1.grid(True)

    plt.show()


def plot_recent_income(
    daily_df: pd.DataFrame,
    num_days: int,
    name_l: list[str],
    color_l: list[str],
):
    # Filter recent days
    daily_df = process_utils.filter_latest_x_rows(daily_df, row_num=num_days)

    fig, ax = plt.subplots(figsize=(18, 6))  # type: ignore

    for name, color in zip(
        name_l,
        color_l,
    ):
        income_l = daily_df[f"{name}_diff"]
        ax.plot_date(
            daily_df.index,
            income_l,  # type: ignore
            fmt="-",
            label=name,
            color=color,
        )

        label_l = [f"{income:,.0f}" for income in income_l]
        utils.add_labels(ax, daily_df.index, income_l, label_l=label_l, color=color)  # type: ignore

    ax.set_title("Daily Income")
    ax.grid(True)
    ax.legend()

    plt.show()


def plot_daily_diff_p_distribution(
    log_df: pd.DataFrame,
    acc_name: str,
    index_name: str,
):
    fig, (ax1, ax2) = plt.subplots(
        figsize=(16, 5),
        ncols=2,
        gridspec_kw={"width_ratios": [3, 1]},
    )  # type: ignore

    index_diff_p_l = log_df[f"{index_name}_diff_p"] * 100
    acc_diff_p_l = log_df[f"{acc_name}_diff_p"] * 100

    # *** Histogram ***

    bins = np.arange(-5, 5, 0.5).tolist()

    # Index histogram
    index_combined_median = index_diff_p_l.median()
    ax1.hist(
        index_diff_p_l,
        bins=bins,
        edgecolor="black",
        alpha=0.25,
        label=f"{index_name} - median:{index_combined_median:.2f}%",
    )
    index_diff_p_mean = index_diff_p_l.mean()  # type: ignore

    ax1.axvline(
        index_diff_p_mean,
        color="blue",
        linestyle="--",
        label="index mean = %.2f%%" % index_diff_p_mean,
        alpha=0.25,
    )

    # acc histogram
    acc_combined_median = acc_diff_p_l.median()
    ax1.hist(
        acc_diff_p_l,
        bins=bins,
        edgecolor="black",
        alpha=0.5,
        label=f"{acc_name} - median:{acc_combined_median:.2f}%",
    )
    acc_diff_p_mean = acc_diff_p_l.mean()  # type: ignore
    ax1.axvline(
        acc_diff_p_mean,
        color="red",
        label="acc mean = %.2f%%" % acc_diff_p_mean,
    )

    ax1.set_title("Distribution of change of combined acc and index %")
    ax1.legend()

    # *** Win / Lose pie chart ***
    win_num = (acc_diff_p_l > index_diff_p_l).sum()
    lose_num = (acc_diff_p_l <= index_diff_p_l).sum()
    ax2.pie(
        [win_num, lose_num],
        labels=[f"Win: {win_num}", f"Lose: {lose_num}"],
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.05, 0],
    )
    ax2.set_title("Win/Lose")

    plt.show()


def plot_growth_xirr_over_time(
    growth_xirr_df: pd.DataFrame,
    anchor_date: str,
    main_acc_name: str,
    main_index_name: str,
    secondary_acc_name_l: list[str],
    secondary_index_name_l: list[str],
):
    fig, (ax1, ax2) = plt.subplots(figsize=(26, 6), nrows=2)  # type: ignore

    def _plot_overall_growth_xirr(
        ax: plt.Axes,
        growth_xirr_df: pd.DataFrame,
        type: str,
    ):
        # Normalize growth from 0
        if type == "growth":
            growth_xirr_df = growth_xirr_df.applymap(lambda e: e - 100)

        for combined_name in [main_acc_name, main_index_name]:
            ax.plot_date(
                growth_xirr_df.index,
                growth_xirr_df[f"{combined_name}_{type}"],
                fmt="-",
                label=combined_name,
            )
            ax.text(
                growth_xirr_df.index[-1],
                growth_xirr_df[f"{combined_name}_{type}"].iloc[-1],  # type: ignore
                f"{growth_xirr_df[f'{combined_name}_{type}'].iloc[-1]:.1f}",  # type: ignore
            )

        for single_name in secondary_acc_name_l:
            ax.plot_date(
                growth_xirr_df.index,
                growth_xirr_df[f"{single_name}_{type}"],
                fmt="-.",
                alpha=0.7,
                label=single_name,
            )

        for single_name in secondary_index_name_l:
            ax.plot_date(
                growth_xirr_df.index,
                growth_xirr_df[f"{single_name}_{type}"],
                fmt=":",
                alpha=0.7,
                label=single_name,
            )

        ax.set_title(f"{type} from {anchor_date}")
        ax.legend(loc="upper left")
        ax.grid(True)
        fig.autofmt_xdate()

    _plot_overall_growth_xirr(ax1, growth_xirr_df, "growth")
    _plot_overall_growth_xirr(ax2, growth_xirr_df, "xirr")

    plt.show()
