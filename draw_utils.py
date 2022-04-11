import matplotlib.pyplot as plt


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
