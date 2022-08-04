# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import pytest

from calflow_parsing.calflow_graph import pg_methods, _CONVERSION_ERROR
from calflow_parsing.graph import Graph, Infeasible


def test_graph_to_program_with_cycle():
    # create a graph with a cycle
    nodes = {0: '0', 1: '1'}
    arcs = [(0, 'a', 1), (1, 'b', 0)]
    graph = Graph(nodes=nodes, edges=arcs, root=0)
    # test that graph_to_program doesn't infinite loop
    with pytest.raises(Infeasible):
        graph.ordering_bottom_up()
    with pytest.raises(Infeasible):
        pg_methods.graph_to_program(graph)
    assert pg_methods.graph_to_lispress(graph) == _CONVERSION_ERROR


def test_graph_to_lispress():
    # create a graph without a cycle, but with a reentrancy
    nodes = {0: 'MyFunction', 1: 'Today'}
    arcs = [(0, ':a', 1), (0, ':b', 1)]
    graph = Graph(nodes=nodes, edges=arcs, root=0)
    assert graph.ordering_bottom_up() == [1, 0]
    expected = "(let (x0 (Today)) (MyFunction :a x0 :b x0))"
    actual = pg_methods.graph_to_lispress(graph)
    assert(actual == expected)
