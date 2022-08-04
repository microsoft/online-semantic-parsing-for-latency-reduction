# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Example usage:

    python -m calflow_parsing.apis \
      --train-dialogues-path="/Users/sathomso/code/sm/incremental-interpretation/DATA/smcalflow2.0/train.dataflow_dialogues.jsonl"
"""
import argparse
from collections import defaultdict, Counter
from typing import Container, List, Dict, Counter as CounterTpe, Tuple

import numpy as np
from numpy.linalg import pinv
from tqdm import tqdm

from calflow_parsing.api_constants import CalflowApis
from calflow_parsing.calflow_graph import ProgramGraph
from calflow_parsing.conf import CALFLOW_DIR
from calflow_parsing.graph import Graph
from dataflow.core.dialogue import Dialogue
from dataflow.core.io import load_jsonl_file


def extract_api_subgraphs(graph: Graph, names: Container[str] = CalflowApis().names) -> List[Graph]:
    """Extract subgraphs corresponding to API function calls."""
    if names is None:
        return [
            extract_subgraph(graph, node_id, node_name)
            for node_id, node_name in graph.nodes.items()
        ]

    return [
        extract_subgraph(graph, node_id, node_name)
        for node_id, node_name in graph.nodes.items()
        if node_name in names
    ]


def extract_subgraph(graph: Graph, subgraph_root_id: int, subgraph_root_name: str) -> Graph:
    """Extract a subgraph from a root node."""
    subgraph = Graph(
        nodes={subgraph_root_id: subgraph_root_name},
        root=subgraph_root_id,
    )

    out_edges = defaultdict(list)
    for s, r, t in graph.edges:
        out_edges[s].append((r, t))

    queue = {subgraph_root_id}
    while queue:
        s = queue.pop()
        for r, t in out_edges[s]:
            if t not in subgraph.nodes:
                subgraph.add_node(t, graph.nodes[t])
                queue.add(t)
            subgraph.add_edge((s, r, t))

    return subgraph


def count_api_dependencies(
        plans: List[Graph],
        names: Container[str] = CalflowApis().names,
) -> Tuple[CounterTpe[str], Dict[str, CounterTpe[str]]]:
    """
    Returns two things:
    - a counter of how many times each API call appears in the data
    - a map from API call type `A` to a map from API call type `B` to
      how many times `B` appears as a dependent of `A`.
    """
    counts: CounterTpe[str] = Counter()
    dependencies: Dict[str, CounterTpe[str]] = defaultdict(lambda: Counter())
    for plan in tqdm(plans, desc="counting dependencies"):
        subgraphs = extract_api_subgraphs(plan, names)
        for subgraph in subgraphs:
            root_name = subgraph.nodes.pop(subgraph.root)
            counts[root_name] += 1
            dependencies[root_name].update(name for name in subgraph.nodes.values() if name in names)
    return counts, dependencies


def solve_for_exclusive_timings(
        counts: CounterTpe[str],
        dependencies: Dict[str, CounterTpe[str]],
        names_and_latencies: Tuple[Tuple[str, float], ...] = CalflowApis().names_and_latencies,
        names: Container[str] = CalflowApis().names,
) -> List[Tuple[str, float]]:
    """
    Assuming that the latencies in `names_and_latencies` *include* all
    dependent API calls, solve for what the latencies must be when
    *excluding* the time of dependent API calls.
    NB: running this showed that the assumption was wrong; `API_P50` latencies
    are already exclusive.
    """
    names = [name for name, _ in names_and_latencies]
    # calculate total latency in ms for each call type
    counts_array = np.array([counts[name] for name in names])
    median_ms = np.array([ms for _, ms in names_and_latencies])
    total_ms = median_ms * counts_array
    # turn counts and deps into a square matrix, so we can solve
    n = len(names)
    deps_array = np.zeros((n, n))
    for i, parent in enumerate(names):
        child_counts = dependencies[parent]
        for j, child in enumerate(names):
            deps_array[i, j] += child_counts[child]
    # always include yourself
    deps_array += np.diag(counts_array)
    # solve `deps_array @ x = total_ms` for `x`.
    # i.e. what must the exclusive time be in order to add up to the total
    # inclusive times?
    x = pinv(deps_array) @ total_ms
    return list(zip(names, x))


def main(train_dialogues_path: str):
    lispress_to_graph = ProgramGraph(type_args_coversion='atomic').lispress_to_graph
    dialogues = list(load_jsonl_file(train_dialogues_path, Dialogue))
    graphs = [
        lispress_to_graph(turn.lispress)
        for dialogue in dialogues
        for turn in dialogue.turns
        if not turn.skip
    ]
    counts, dependencies = count_api_dependencies(graphs)
    print("Counts:")
    for parent, children in sorted(dependencies.items()):
        print()
        print(parent, counts[parent])
        for child, count in sorted(children.items()):
            print("->", child, count)

    print()
    print("Solved timings:")
    exclusive_timings = solve_for_exclusive_timings(counts, dependencies)
    for name, ms in exclusive_timings:
        print(name, ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dialogues-path",
        type=str,
        help="jsonl file with training dialogues",
        default=CALFLOW_DIR / "train.dataflow_dialogues.jsonl"
    )
    args = parser.parse_args()
    main(args.train_dialogues_path)
