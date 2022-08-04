# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from calflow_parsing.apis import extract_subgraph
from calflow_parsing.calflow_graph import lispress_to_graph, graph_to_lispress


def test_extract_subgraph():
    """Regression test"""
    lispress = """(Yield (Execute (ReviseConstraint (refer (^(Dynamic) roleConstraint (Path.apply "output"))) (^(Event) ConstraintTypeIntension) (Event.showAs_? (?= (ShowAsStatus.OutOfOffice))))))"""
    graph = lispress_to_graph(lispress)
    sg = extract_subgraph(graph, graph.root, graph.nodes[graph.root])
    assert sg.is_identical(graph)
    assert graph_to_lispress(sg) == graph_to_lispress(graph)
