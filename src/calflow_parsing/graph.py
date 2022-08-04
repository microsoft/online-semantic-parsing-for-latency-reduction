# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Generic graph definition."""
import copy
from typing import Dict, Tuple, List, Optional


class Infeasible(Exception):
    pass


class Graph:
    def __init__(
            self,
            nodes: Dict[int, str] = None,
            edges: Optional[List[Tuple[int, str, int]]] = None,
            root: Optional[int] = None,
    ):
        self.nodes: Dict[int, str] = nodes or {}
        self.edges: List[Tuple[int, str, int]] = edges or []
        self.root: Optional[int] = root

    @property
    def node_ids(self):
        return list(self.nodes.keys())

    @property
    def next_node_id(self):
        if self.nodes:
            return max(self.node_ids) + 1
        else:
            return 1    # 0 is reserved for special ROOT node (if any)

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        return len(self.edges)

    @property
    def out_edges(self) -> Dict[int, Dict[int, List[str]]]:
        """Map from s -> t -> [r], for all arcs (s, r, t)"""
        result = {}
        for s, r, t in self.edges:
            result.setdefault(s, {}).setdefault(t, []).append(r)
        return result

    @property
    def in_edges(self) -> Dict[int, Dict[int, List[str]]]:
        """Map from t -> s -> [r], for all arcs (s, r, t)"""
        result = {}
        for s, r, t in self.edges:
            result.setdefault(t, {}).setdefault(s, []).append(r)
        return result

    def sanity_check(self):
        """check if the graph is valid."""
        for nid, node_name in self.nodes.items():
            assert isinstance(nid, int), 'node_id is not of type int'
            assert isinstance(node_name, str), 'node_name is not of type str'
        for s, r, t in self.edges:
            assert s in self.nodes and t in self.nodes, f'node_id {s} or {t} for edges not in node_id list'
        if self.root is not None:
            assert self.root in self.nodes, f'root id {self.root} not in node_id list'

    def add_node(self, node_id: int = None, node_name: str = None):
        node_id = node_id or self.next_node_id  # when node_id is not provided, default to the next one
        assert node_id not in self.node_ids
        self.nodes[node_id] = node_name

    def add_edge(self, edge: Tuple[int, str, int]):
        assert edge is not None
        assert edge[0] in self.node_ids and edge[2] in self.node_ids, f'edge {edge} is invalid'
        self.edges.append(edge)

    def merge(self, nodes: Dict[int, str] = None, edges: List[Tuple[int, str, int]] = None):
        """Merge another subgraph (could be only new nodes, new edges) into the current graph.
        They should be compatible (e.g. node numbering not overlapping).
        """
        assert not set(self.node_ids).intersection(set(nodes.keys())), 'node ids overlapping; can not merge'

        self.nodes.update(nodes)    # return value is None
        self.edges += edges
        # NOTE this should not change the root of the current graph
        # TODO add a check that the new edges do not incur any parent of the current root

    def is_identical(self, graph: 'Graph') -> bool:
        """check if two graphs are identical.
        Here it means:
        - the same number of nodes and edges
        - all node names are matched
        - all edge triplets are matched
        """
        if self.num_nodes != graph.num_nodes:
            return False
        if self.num_edges != graph.num_edges:
            return False
        # compare the node names
        nodes2 = copy.deepcopy(graph.nodes)
        for nid, nn in self.nodes.items():
            for nid2, nn2 in list(nodes2.items()):
                if nn == nn2:
                    # one matched name found
                    del nodes2[nid2]
                    break
            else:
                # no matched name found in the remaining nodes
                return False
        # compare the edge triplets
        edges2 = copy.deepcopy(graph.edges)
        for s, r, t in self.edges:
            for i, (s2, r2, t2) in enumerate(edges2):
                if r == r2 and self.nodes[s] == graph.nodes[s2] and self.nodes[t] == graph.nodes[t2]:
                    # one matched edge triplet found
                    del edges2[i]
                    break
            else:
                # no matched edge triplet found in the remaining edges
                return False

        return True

    def ordering_bottom_up(self) -> List[int]:
        """Order the nodes with bottom-up traversal.
        This is used, e.g. in converting the graph into a program, where the expressions follow the bottom-up order
        to be inserted.

        Returns:
            ordered_node_ids (List[int]): node ids with the desired order
        """
        out_edges = self.out_edges
        in_edges = self.in_edges
        result = []
        queue = set(self.node_ids)
        while leaves := {n for n in queue if len(out_edges.get(n, {})) == 0}:
            result += list(leaves)
            # remove leaves from the queue
            queue.difference_update(leaves)
            # remove all edges incoming to t
            for t in leaves:
                for s in in_edges.pop(t, {}).keys():
                    s_out_edges = out_edges.get(s, {})
                    if t in s_out_edges:
                        del s_out_edges[t]
        if queue:
            raise Infeasible("Graph contains a cycle: {}".format(self))
        return result

    def __str__(self):
        # TODO add more control for AMR-like penman format output (possibly in another function), also to enable
        #      Smatch evaluation between partial graphs
        output = ''
        # nodes
        for nid, name in self.nodes.items():
            output += f'# ::node\t{nid}\t{name}' + '\n'
        # root
        if self.root is not None:
            output += f'# ::root\t{self.root}\t{self.nodes[self.root]}' + '\n'
        else:
            output += f'# ::root\t{"<unspecified>"}' + '\n'
        # edges
        for s, r, t in self.edges:
            s_name = self.nodes[s]
            t_name = self.nodes[t]
            output += f'# ::edge\t{s_name}\t{r}\t{t_name}\t{s}\t{t}' + '\n'

        return output
