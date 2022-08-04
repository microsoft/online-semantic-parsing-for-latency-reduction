# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
from copy import deepcopy
from typing import List, Tuple
from dataclasses import dataclass, field

from tqdm import tqdm

from calflow_parsing.graph import Graph
from calflow_parsing.calflow_graph import actions_to_graph
from calflow_parsing.io import read_string_sentences
from calflow_parsing.exact_match import anonymize_strings_in_graph
from calflow_parsing.api_constants import CalflowApis, TreedstApis


def extract_api_subgraphs(graph: Graph, keynode_list: List[str]) -> Tuple[List[Graph], List[str]]:
    """Extract subgraphs corresponding to API function calls."""
    subgraphs = []
    subgraph_root_ids = []
    subgraph_root_names = []
    for node_id, node_name in graph.nodes.items():
        if node_id not in subgraph_root_ids and node_name in keynode_list:
            # record the node ids that have been used
            subgraph_root_ids.append(node_id)
            subgraph_root_names.append(node_name)

            # extract the subgraph
            subgraph = extract_subgraph(graph, node_id, node_name)
            subgraphs.append(subgraph)

    return subgraphs, subgraph_root_names


def extract_subgraph(graph: Graph, subgraph_root_id: int, subgraph_root_name: str) -> Graph:
    """Extract a subgraph from a root node."""
    subgraph = Graph(nodes={subgraph_root_id: subgraph_root_name}, root=subgraph_root_id)

    last_nodes = {subgraph_root_id}
    current_nodes = set()
    remaining_edges = deepcopy(graph.edges)

    # breakpoint()

    while True:
        for i, (s, r, t) in enumerate(remaining_edges):
            if s in last_nodes:
                if t not in subgraph.nodes:
                    # add node
                    subgraph.add_node(t, graph.nodes[t])
                    current_nodes.add(t)
                # add edge
                subgraph.add_edge((s, r, t))
                # remove the edge
                del remaining_edges[i]
        if current_nodes:
            last_nodes = last_nodes.union(current_nodes)
            current_nodes = set()
        else:
            # empty new node list
            break

    # breakpoint()

    return subgraph


def graph_api_subgraph_match(gold_graph: Graph,
                             test_graph: Graph,
                             keynode_list: List[str]) -> Tuple[int, int, int, float, float, float]:
    """Match certain subgraphs (e.g. API function calls) between two graphs, and compute the precision, recall,
    and f1 scores.
    The subgraphs are defined based on a keyword list.

    Args:
        gold_graph (Graph): [description]
        test_graph (Graph): [description]

    Returns:
        Tuple[float, float, float]: [description]
    """
    gold_subgraphs, gold_subgraph_roots = extract_api_subgraphs(gold_graph, keynode_list)
    test_subgraphs, test_subgraph_roots = extract_api_subgraphs(test_graph, keynode_list)

    # compare the subgraphs
    num_gold_subgraphs = len(gold_subgraphs)
    num_test_subgraphs = len(test_subgraphs)
    num_matched = 0

    if num_gold_subgraphs >= 1:

        for gold_subgraph in gold_subgraphs:
            for test_subgraph in test_subgraphs:
                if gold_subgraph.is_identical(test_subgraph):
                    num_matched += 1
                    break

        precision = num_matched / (num_test_subgraphs + 1e-6)
        recall = num_matched / (num_gold_subgraphs)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

    else:

        # no gold subgraph
        precision = -1
        recall = -1
        f1 = -1

    return num_gold_subgraphs, num_test_subgraphs, num_matched, precision, recall, f1


@dataclass
class GraphAPISubGraphMatchScores:
    num_total_examples: int = field(default=0, metadata={'annotation': 'total number of examples evaluated'})
    num_matched_graphs: int = field(default=0, metadata={'annotation': 'number of examples with graph match'})
    num_matched_graphs_anonym_string: int = field(
        default=0,
        metadata={'annotation': 'number of examples with graph match with strings anonymized'}
    )
    num_apis_list: List = field(
        default=None,
        metadata={'annotation': 'list of number of API subgraphs for each graph'}
    )
    num_graphs_with_apis: int = field(default=0, metadata={'annotation': 'total number of graphs with api subgraphs'})
    precision_list: List = field(
        default=None,
        metadata={'annotation': 'list of precision values for each graph evaluation'}
        )
    recall_list: List = field(
        default=None,
        metadata={'annotation': 'list of recall values for each graph evaluation'}
        )
    f1_list: List = field(
        default=None,
        metadata={'annotation': 'list of f1 values for each graph evaluation'}
        )
    precision_anonym_string_list: List = field(
        default=None,
        metadata={'annotation': 'list of precision values for each graph evaluation with strings anonymized'}
        )
    recall_anonym_string_list: List = field(
        default=None,
        metadata={'annotation': 'list of recall values for each graph evaluation with strings anonymized'}
        )
    f1_anonym_string_list: List = field(
        default=None,
        metadata={'annotation': 'list of f1 values for each graph evaluation with strings anonymized'}
        )

    @property
    def accuracy(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return self.num_matched_graphs / self.num_total_examples

    @property
    def accuracy_graph_strings_anonymized(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return self.num_matched_graphs_anonym_string / self.num_total_examples

    @property
    def average_num_apis(self) -> int:
        if self.num_graphs_with_apis == 0 :
            return 0
        else:
            return sum(self.num_apis_list) / self.num_graphs_with_apis

    @property
    def precision(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum([x for x in self.precision_list if x != -1]) / self.num_graphs_with_apis
        # return sum(self.precision_list) / self.num_total_examples

    @property
    def recall(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum([x for x in self.recall_list if x != -1]) / self.num_graphs_with_apis
        # return sum(self.recall_list) / self.num_total_examples

    @property
    def f1(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum([x for x in self.f1_list if x != -1]) / self.num_graphs_with_apis
        # return sum(self.f1_list) / self.num_total_examples

    @property
    def precision_anonym_string(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum([x for x in self.precision_anonym_string_list if x != -1]) / self.num_graphs_with_apis
        # return sum(self.precision_anonym_string_list) / self.num_total_examples

    @property
    def recall_anonym_string(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum([x for x in self.recall_anonym_string_list if x != -1]) / self.num_graphs_with_apis
        # return sum(self.recall_anonym_string_list) / self.num_total_examples

    @property
    def f1_anonym_string(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum([x for x in self.f1_anonym_string_list if x != -1]) / self.num_graphs_with_apis
        # return sum(self.f1_anonym_string_list) / self.num_total_examples

    def add(self, ref_graph: Graph, pred_graph: Graph, keynode_list: List[str]):

        if ref_graph.is_identical(pred_graph):
            self.num_matched_graphs += 1

        # test graph match with strings all anonymized (to see quality of string generation or copying)
        if anonymize_strings_in_graph(ref_graph).is_identical(anonymize_strings_in_graph(pred_graph)):
            self.num_matched_graphs_anonym_string += 1

        self.num_total_examples += 1

        # API subgraph match scores
        num_ref_subgraphs, num_pred_subgraphs, num_matched, precision, recall, f1 = graph_api_subgraph_match(
            ref_graph, pred_graph, keynode_list)

        if self.num_apis_list is not None:
            self.num_apis_list.append(num_ref_subgraphs)
        else:
            self.num_apis_list = [num_ref_subgraphs]

        if num_ref_subgraphs >= 1:
            self.num_graphs_with_apis += 1


        # record precision, recall, and f1
        if self.precision_list is not None:
            self.precision_list.append(precision)
        else:
            self.precision_list = [precision]

        if self.recall_list is not None:
            self.recall_list.append(recall)
        else:
            self.recall_list = [recall]

        if self.f1_list is not None:
            self.f1_list.append(f1)
        else:
            self.f1_list = [f1]

        # tuple match scores with strings all anonymized
        _, _, _, precision_anonym, recall_anonym, f1_anonym = \
            graph_api_subgraph_match(anonymize_strings_in_graph(ref_graph),
                                     anonymize_strings_in_graph(pred_graph),
                                     keynode_list,
                                     )
        if self.precision_anonym_string_list is not None:
            self.precision_anonym_string_list.append(precision_anonym)
        else:
            self.precision_anonym_string_list = [precision_anonym]

        if self.recall_anonym_string_list is not None:
            self.recall_anonym_string_list.append(recall_anonym)
        else:
            self.recall_anonym_string_list = [recall_anonym]

        if self.f1_anonym_string_list is not None:
            self.f1_anonym_string_list.append(f1_anonym)
        else:
            self.f1_anonym_string_list = [f1_anonym]

    def __str__(self):
        output = ''
        output += f'num_total_examples: {self.num_total_examples}' + '\n'
        output += f'num_matched_graphs: {self.num_matched_graphs}' + '\n'
        output += f'num_matched_graphs_anonym_string: {self.num_matched_graphs_anonym_string}' + '\n'
        # exact match of graph
        output += f'exact_match_graphs: {self.accuracy:.5f}' + '\n'
        output += f'exact_match_graph_strings_anonymized: {self.accuracy_graph_strings_anonymized:.5f}' + '\n'
        # number of api subgraphs
        output += f'num_graphs_with_apis: {self.num_graphs_with_apis}' + '\n'
        output += f'num_apis_average: {self.average_num_apis:.5f}' + '\n'
        # api subgraph match
        output += f'api_match_average_precision: {self.precision:.5f}' + '\n'
        output += f'api_match_average_recall: {self.recall:.5f}' + '\n'
        output += f'api_match_average_f1: {self.f1:.5f}' + '\n'
        # api subgraph match with anonymized strings
        output += f'api_match_strings_anonymized_average_precision: {self.precision_anonym_string:.5f}' + '\n'
        output += f'api_match_strings_anonymized_average_recall: {self.recall_anonym_string:.5f}' + '\n'
        output += f'api_match_strings_anonymized_average_f1: {self.f1_anonym_string:.5f}'

        return output


def evaluate_graph_api_subgraphs(
    test_actions_file: str,
    gold_actions_file: str,
    keynode_list: List[str],
    ) -> GraphAPISubGraphMatchScores:
    """Evaluate generated graph (represented by actions) with tuple match metric
    (including precision, recall, and F1).

    Args:
        test_actions_file (str): [description]
        gold_actions_file (str, optional): [description]. Defaults to None.

    Returns:
        GraphAPISubGraphMatchScores: [description]
    """
    test_actions = read_string_sentences(test_actions_file)
    gold_actions = read_string_sentences(gold_actions_file)

    graph_tuple_match_scores = GraphAPISubGraphMatchScores()

    for idx, (gold_act, test_act) in tqdm(enumerate(zip(gold_actions, test_actions)),
                                          unit=' graphs (action sequences)', desc='Evaluation - Graph API Match'):
        gold_graph = actions_to_graph(gold_act.split(' '))
        test_graph = actions_to_graph(test_act.split(' '))

        graph_tuple_match_scores.add(
            ref_graph=gold_graph,
            pred_graph=test_graph,
            keynode_list=keynode_list,
            )

    return graph_tuple_match_scores


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of Graph Tuple Match')
    parser.add_argument('--test_actions', type=str,
                        help='input file with the testing graph actions')
    parser.add_argument('--gold_actions', type=str,
                        help='input file with the gold graph actions')
    parser.add_argument('--out_file', type=str,
                        help='output file to store the resulted tuple match scores')
    parser.add_argument('--dataset', type=str, default='smcalflow', choices=['smcalflow', 'treedst'],
                        help='dataset that we are dealing with')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    keynode_list = CalflowApis().names if args.dataset == 'smcalflow' else TreedstApis().names

    graph_tuple_match_scores = evaluate_graph_api_subgraphs(
        test_actions_file=args.test_actions,
        gold_actions_file=args.gold_actions,
        keynode_list=keynode_list,
    )

    with open(args.out_file, 'w') as f:
        print(graph_tuple_match_scores, file=f)
        print(f'Evaluation - Graph API Subgraphs Match - results saved to {args.out_file}')
