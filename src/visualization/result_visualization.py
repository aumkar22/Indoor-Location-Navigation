import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from PIL import Image

from typing import NoReturn, List, Tuple


def trace_fix(trajectory: np.ndarray, colour_str: str) -> Tuple[List, ...]:

    """

    :param trajectory:
    :param colour_str:
    :return:
    """

    size_list = [6] * trajectory.shape[0]
    size_list[0] = 10
    size_list[-1] = 10

    color_list = [colour_str] * trajectory.shape[0]
    color_list[0] = colour_str
    color_list[-1] = colour_str

    position_count = {}
    text_list = []
    for i in range(trajectory.shape[0]):
        if str(trajectory[i]) in position_count:
            position_count[str(trajectory[i])] += 1
        else:
            position_count[str(trajectory[i])] = 0
        text_list.append("        " * position_count[str(trajectory[i])] + f"{i}")
    text_list[0] = "Start Point: 0"
    text_list[-1] = f"End Point: {trajectory.shape[0] - 1}"

    return size_list, color_list, text_list


def visualize_trajectory(
    trajectory,
    estimated_way,
    floor_plan_filename,
    width_meter,
    height_meter,
    title=None,
    mode="lines + markers + text",
    show=False,
):

    """
    Original function provided by hosts of the competition and found at:
    https://github.com/location-competition/indoor-location-competition-20/blob/master/visualize_f.py

    This is a modified version of the original.

    :param trajectory:
    :param estimated_way:
    :param floor_plan_filename:
    :param width_meter:
    :param height_meter:
    :param title:
    :param mode:
    :param show:
    :return:
    """
    fig = go.Figure()

    # add trajectory

    size_list, color_list, text_list = trace_fix(trajectory, "rgba(4, 174, 4, 0.5)")

    size_list_estimated, color_list_estimated, text_list_estimated = trace_fix(
        estimated_way, "rgba(147,112,219,0.5)"
    )

    fig.add_trace(
        go.Scattergl(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            mode=mode,
            marker=dict(size=size_list, color=color_list),
            line=dict(shape="linear", color="rgb(100, 10, 100)", width=2, dash="dot"),
            text=text_list,
            textposition="top center",
            name="trajectory",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=estimated_way[:, 0],
            y=estimated_way[:, 1],
            mode=mode,
            marker=dict(size=size_list_estimated, color=color_list_estimated),
            line=dict(shape="linear", color="rgb(23, 57, 211)", width=2, dash="dot"),
            text=text_list_estimated,
            textposition="top center",
            name="estimated_trajectory",
        )
    )

    # add floor plan
    floor_plan = Image.open(floor_plan_filename)
    fig.update_layout(
        images=[
            go.layout.Image(
                source=floor_plan,
                xref="x",
                yref="y",
                x=0,
                y=height_meter,
                sizex=width_meter,
                sizey=height_meter,
                sizing="contain",
                opacity=1,
                layer="below",
            )
        ]
    )

    # configure
    fig.update_xaxes(autorange=False, range=[0, width_meter])
    fig.update_yaxes(autorange=False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=go.layout.Title(text=title or "No title.", xref="paper", x=0,),
        autosize=True,
        width=900,
        height=200 + 900 * height_meter / width_meter,
        template="plotly_white",
    )

    if show:
        fig.show()

    return fig
