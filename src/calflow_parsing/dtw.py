# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""DTW algorithm with general inputs and distance metrics.
"""
from collections import defaultdict
from typing import List, Any, Callable, Tuple, Union


def dtw(x: List[Any], y: List[Any], dist: Callable) -> Tuple[Union[int, float], List[Tuple[int, int]]]:
    """DTW algorithm to find the optimal match between two list of elements.

    Args:
        x (List[Any]): [description]
        y (List[Any]): [description]
        dist (Callable): [description]

    Returns:
        Tuple[Union[int, float], List[Tuple[int, int]]]: [description]
    """
    len_x = len(x)
    len_y = len(y)

    dtw_matrix = defaultdict(lambda: (float('inf'),))
    dtw_matrix[0, 0] = (0, 0, 0)

    for i in range(1, len_x + 1):
        for j in range(1, len_y + 1):
            dt = dist(x[i - 1], y[j - 1])
            # breakpoint()
            dtw_matrix[i, j] = min(
                (dtw_matrix[i, j - 1][0] + dt, i, j - 1),
                (dtw_matrix[i - 1, j - 1][0] + dt, i - 1, j - 1),
                (dtw_matrix[i - 1, j][0] + dt, i - 1, j),
            )

    dtw_cost = dtw_matrix[len_x, len_y][0]

    # get the path
    path = []
    i = len_x
    j = len_y
    while not (i == j == 0):
        path.append([i - 1, j - 1])    # NOTE the index in the dtw_matrix starts from 1, while in path it starts from 0
        i, j = dtw_matrix[i, j][1], dtw_matrix[i, j][2]

    path.reverse()

    return dtw_cost, path
