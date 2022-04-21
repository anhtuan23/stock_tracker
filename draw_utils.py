import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils


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
        if profit > 0:
            profit_label = f"Profit:{profit:,}"
            invest_label = f"Invest:{invest:,}"
            ax.pie(
                [profit, invest], labels=[profit_label, invest_label], autopct="%1.1f%%"
            )

            nav = profit + invest
            ax.set_title(f"{name}:{nav:,}")
