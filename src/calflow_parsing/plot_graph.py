# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from collections import OrderedDict
from operator import itemgetter as ig
from typing import Union, List

import matplotlib.pyplot as plt
import networkx as nx

from calflow_parsing.graph import Graph


def dag_layout(g, edge_orders):
    """
    2d layout that works okay for our plans, which are DAGs that are mostly
    trees with maybe a few reentrancies.
    """
    pos = {}
    layers = get_layers(g)
    for i, layer in enumerate(layers):
        y = -i
        dx = 1 / float(len(layer))

        def min_parent_x(n):
            parent_xs = [
                pos[e[0]][0] + dx * (edge_orders[e] - 0.5)
                for e in g.in_edges(n)
                if e[0] in pos
            ]
            # return sum(parent_xs) / float(len(parent_xs)) if parent_xs else 0.0
            return min(parent_xs) if parent_xs else 0.0

        layer = sorted(layer, key=min_parent_x)
        for j, n in enumerate(layer):
            x = min_parent_x(n) + dx * (j + 0.5) - 0.5
            pos[n] = (x, y)
    return pos


def get_layers(g):
    """
    Groups nodes of a DAG into layers so that each node is at least 1 layer below
    all of its ancestors.
    If `g` is not a DAG, makes an effort to minimize upward arcs.
    """
    # pylint: disable=C1801
    if len(g) == 0:
        return []
    in_degrees = OrderedDict(g.in_degree)
    layers_dict = {}
    while in_degrees:
        node, _ = min(in_degrees.items(), key=ig(1))
        del in_degrees[node]
        layer = (
            max([-1] + [layers_dict.get(parent, -1) for parent, _ in g.in_edges(node)])
            + 1
        )
        layers_dict[node] = layer
        for _, child in g.out_edges(node):
            in_degrees[child] -= 1
    max_layer = max(layers_dict.values())
    layers = [[] for _ in range(max_layer + 1)]
    for n, i in layers_dict.items():
        layers[i].append(n)
    return layers


def get_out_edge_orders(graph):
    """For each node, sort its out edges based on positional args and named args, with the latter
    sorted alphabatically.
    """
    children = {}
    for s, r, t in graph.edges:
        children.setdefault(s, []).append((r, t))

    edge_orders = {}
    for nid, out_list in children.items():
        rs, ts = zip(*out_list)
        sorted_args, sorted_idx = sort_args(rs)
        for i, idx in enumerate(sorted_idx):
            edge_orders[(nid, ts[idx])] = i

    return edge_orders


def sort_args(args):
    """Sort argument names: type args (':type-arg{i}'), positional args (':arg{i}'), then named
    args (sorting with alphabatical order)."""
    type_args = []
    pos_args = []
    named_args = []
    for idx, label in enumerate(args):
        if label.startswith(':type'):
            type_args.append((label, idx))
        elif label.startswith(':arg'):
            pos_args.append((label, idx))
        else:
            named_args.append((label, idx))
    sorted_items = sorted(type_args) + sorted(pos_args) + sorted(named_args)
    sorted_args, sorted_idx = zip(*sorted_items)
    return sorted_args, sorted_idx


def plot_graph(graph: Graph):
    """Plot a graph with DAG layout, with both node and edge labels.

    Args:
        graph (Graph): [description]

    Returns:
        [type]: [description]
    """
    G = nx.DiGraph()
    G.add_nodes_from(graph.nodes)
    G.add_edges_from([(s, t) for (s, r, t) in graph.edges])

    node_labels = graph.nodes
    edge_labels = {(s, t): r for (s, r, t) in graph.edges}

    plt.figure(figsize=(16, 10))

    # pos = nx.spring_layout(G)

    edge_orders = get_out_edge_orders(graph)
    pos = dag_layout(G, edge_orders)

    # pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')

    nx.draw_networkx(G, pos,
                     labels=node_labels, with_labels=True,
                     node_size=5,
                     font_size=10,
                     edge_color="lightgrey",    # "lightblue"
                    )
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=edge_labels,
                                 font_size=8,
                                 font_color="red",
                                )

    plt.show()

    return G


def plot_graph_from_actions(actions: Union[str, List[str]]):
    """Get the graph from an action sequence, and plot the graph.

    Args:
        actions (Union[str, List[str]]): [description]
    """
    from calflow_parsing.calflow_graph import actions_to_graph

    if isinstance(actions, str):
        actions = actions.split()
    elif isinstance(actions, list):
        ...
    else:
        raise TypeError

    graph = actions_to_graph(actions)
    g = plot_graph(graph)

    return graph, g


def plot_graph_from_lispress(lispress: str):
    """Get the graph from a lispress.

    Args:
        lispress (str): lispress string
    """
    from calflow_parsing.calflow_graph import ProgramGraph

    program_graph_methods = ProgramGraph(type_args_coversion='atomic')
    graph = program_graph_methods.lispress_to_graph(lispress)

    g = plot_graph(graph)

    return graph, g
