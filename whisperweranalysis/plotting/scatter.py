from itertools import cycle
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .line import adjustFigAspect, validate_input

CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"
Cobalt_Blue = "#0047AB"
Orange = "#FFA500"

color_list = [
    CB91_Blue,
    CB91_Pink,
    CB91_Green,
    CB91_Amber,
    CB91_Purple,
    CB91_Violet,
    Cobalt_Blue,
    Orange,
]


CB91_Grad_BP = [
    "#2cbdfe",
    "#2fb9fc",
    "#33b4fa",
    "#36b0f8",
    "#3aacf6",
    "#3da8f4",
    "#41a3f2",
    "#449ff0",
    "#489bee",
    "#4b97ec",
    "#4f92ea",
    "#528ee8",
    "#568ae6",
    "#5986e4",
    "#5c81e2",
    "#607de0",
    "#6379de",
    "#6775dc",
    "#6a70da",
    "#6e6cd8",
    "#7168d7",
    "#7564d5",
    "#785fd3",
    "#7c5bd1",
    "#7f57cf",
    "#8353cd",
    "#864ecb",
    "#894ac9",
    "#8d46c7",
    "#9042c5",
    "#943dc3",
    "#9739c1",
    "#9b35bf",
    "#9e31bd",
    "#a22cbb",
    "#a528b9",
    "#a924b7",
    "#ac20b5",
    "#b01bb3",
    "#b317b1",
]

sns.set(
    font="CMU Sans Serif",
    rc={
        # 'axes.axisbelow': False,
        # 'axes.edgecolor': 'lightgrey',
        # 'axes.facecolor': 'None',
        # 'axes.grid': False,
        # 'axes.labelcolor': 'dimgrey',
        # 'axes.spines.right': False,
        # 'axes.spines.top': False,
        # 'figure.facecolor': 'white',
        # 'lines.solid_capstyle': 'round',
        # 'patch.edgecolor': 'w',
        # 'patch.force_edgecolor': True,
        # 'text.color': 'dimgrey',
        # 'xtick.bottom': False,
        # 'xtick.color': 'dimgrey',
        # 'xtick.direction': 'out',
        # 'xtick.top': False,
        # 'ytick.color': 'dimgrey',
        # 'ytick.direction': 'out',
        # 'ytick.left': False,
        # 'ytick.right': False,
        "font.family": "sans-serif",
        "mathtext.fontset": "cm",
        "axes.prop_cycle": plt.cycler(color=color_list),
    },
)

# sns.set_context("notebook", rc={"font.size": 16,
#                                 "axes.titlesize": 20,
#                                 "axes.labelsize": 18})


def plot_scatter_from_list_of_dict(
    xs: List[List[int]],
    ys: List[List[float]],
    x_label: str,
    y_label: str,
    legend_label: List[str],
    title: Optional[str] = None,
    x_ticks: Optional[List[Any]] = [],
    x_tick_labels: Optional[List[Any]] = [],
    y_ticks: Optional[List[Any]] = [],
    y_tick_labels: Optional[List[Any]] = [],
    markers: Optional[List[str]] = [],
    plot_colors: Optional[List[str]] = [],
    file_name: Optional[str] = None,
    xlim: Optional[Tuple[float]] = None,
    ylim: Optional[Tuple[float]] = None,
    aspect_ratio: Optional[Tuple[float]] = (1, 1),
    plot_log_x: bool = False,
    plot_log_y: bool = False,
    radius: float = 1,
    alpha: float = 0.2,
    linestyle: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Plots a line graph with multiple lines.

    If you get font errors:
    install:
        conda install -c conda-forge mscorefonts
    then:
        import matplotlib; matplotlib.get_cachedir()  # Delete this



    Args:
        x (List[Any]): X axis values.
        If only one input is provided it will be used for all lines by broadcasting to the length of y.
        y (List[Any]): Y axis values.
        x_label (str): X axis label.
        y_label (str): Y axis label.
        title (str): Title of the plot.
        legend_label (List[str]): List of legend labels.
        x_tick_labels (Optional[List[Any]]): List of x tick labels. Defaults to [].
        line_format (Optional[List[str]], optional): List of line formats like (-, --, :-, : etc). Defaults to None.
        file_name (Optional[str], optional): Name of the file to save the plot to. Defaults to None.
        xlim (Optional[Tuple[float]], optional): limits of the x axis (min, max). Defaults to (0, None).
        ylim (Optional[Tuple[float]], optional): limits of the y axis (min, max). Defaults to (0, None).
        aspect_ratio (Optional[Tuple[float]], optional): aspect ratio of the plot (width, height). Defaults to (1, 1).
        figsize (Optional[Tuple[float]], optional): size of the figure (width, height). Defaults to (6.4, 4.8).
        plot_log_x (bool, optional): Whether to plot the x axis in log scale. Defaults to False.
        plot_log_y (bool, optional): Whether to plot the y axis in log scale. Defaults to False.
    """
    validate_input(xs, list)
    validate_input(xs[0], list)
    validate_input(xs[0][0], int)

    assert len(xs) == len(
        ys
    ), f"input lengths must be of the same length but received x: {len(xs)}, y: {len(ys)}"
    assert len(xs[0]) == len(
        ys[0]
    ), f"per scatter x and y must be of the same length but received x: {len(xs[0])}, y: {len(ys[0])}"
    assert len(xs) == len(
        legend_label
    ), f"dict and legend label must be of the same \
        length but received x: {len(xs)} and legend_label: {len(legend_label)}"

    if not markers:
        markers = ["o" for _ in range(len(xs))]
    elif len(markers) < len(xs):
        raise ValueError(
            f"Markers must be minimum of length {len(xs)} but was {len(markers)}"
        )

    if not plot_colors:
        color_iterator = cycle(color_list)
    else:
        color_iterator = cycle(plot_colors)

    fig, ax = plt.subplots()

    plt.rcParams["font.sans-serif"] = "CMU Sans Serif"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "cm"

    for i, (x, y) in enumerate(zip(xs, ys)):
        color = next(color_iterator)
        ax.scatter(
            x,
            y,
            marker=markers[i],
            color=color,
            s=radius,
            alpha=alpha,
            # label=legend_label[i],
        )
        b, a = np.polyfit(x, y, deg=1)
        xseq = np.linspace(0, 1500, num=100)
        # Plot regression line
        ax.plot(
            xseq,
            a + b * xseq,
            color=color,
            lw=1.5,
            label=legend_label[i],
            linestyle=linestyle[i],
        )

    if plot_log_x:
        ax.set_xscale("log")
    if plot_log_y:
        ax.set_yscale("log")

    if x_tick_labels:
        assert len(x_ticks) == len(
            x_tick_labels
        ), f"x_tick_labels must be of the same length \
            as x_ticks but received x_tick_labels: {len(x_tick_labels)}, x_ticks: {len(x_ticks)}."
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)

    if y_tick_labels:
        assert len(y_ticks) == len(
            y_tick_labels
        ), f"y_tick_labels must be of the  \
            same length as y_ticks but received y_tick_labels: {len(y_tick_labels)}, y_ticks: {len(y_ticks)}."
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

    adjustFigAspect(fig, aspect_ratio[0] / aspect_ratio[1])

    # ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(x_label)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    leg = ax.legend(loc=(1.04, 0.15), markerscale=2, frameon=False)
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    if file_name:
        plt.savefig(file_name, bbox_inches="tight")
    plt.show()
    plt.close()
