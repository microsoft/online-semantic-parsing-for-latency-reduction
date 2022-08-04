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


def graph_tuple_match(gold_graph: Graph, test_graph: Graph) -> Tuple[float, float, float]:
    """Match the tuples between two graphs, and compute the precision, recall, and f1 scores.
    The tuples are defined as the union of the nodes and edges (triplets), to account for isolated nodes.

    Args:
        gold_graph (Graph): [description]
        test_graph (Graph): [description]

    Returns:
        Tuple[float, float, float]: [description]
    """
    # compare the nodes
    num_matched_nodes = 0
    test_nodes = deepcopy(test_graph.nodes)
    for nid, nn in gold_graph.nodes.items():
        for nid2, nn2 in list(test_nodes.items()):
            if nn == nn2:
                # one matched node name found
                num_matched_nodes += 1
                del test_nodes[nid2]
                break

    # compare the edge triplets
    num_matched_triplets = 0
    test_edges = deepcopy(test_graph.edges)
    for s, r, t in gold_graph.edges:
        for i, (s2, r2, t2) in enumerate(test_edges):
            if r == r2 and gold_graph.nodes[s] == test_graph.nodes[s2] and gold_graph.nodes[t] == test_graph.nodes[t2]:
                # one matched edge triplet found
                num_matched_triplets += 1
                del test_edges[i]
                break

    # compute precision, recall, and f1 scores
    num_gold_nodes = gold_graph.num_nodes
    num_gold_triplets = gold_graph.num_edges
    num_test_nodes = test_graph.num_nodes
    num_test_triplets = test_graph.num_edges

    num_gold_tuples = num_gold_nodes + num_gold_triplets
    num_test_tuples = num_test_nodes + num_test_triplets
    num_matched_tuples = num_matched_nodes + num_matched_triplets

    precision = num_matched_tuples / (num_test_tuples + 1e-6)
    recall = num_matched_tuples / (num_gold_tuples + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1


@dataclass
class GraphTupleMatchScores:
    num_total_examples: int = field(default=0, metadata={'annotation': 'total number of examples evaluated'})
    num_matched_graphs: int = field(default=0, metadata={'annotation': 'number of examples with graph match'})
    num_matched_graphs_anonym_string: int = field(
        default=0,
        metadata={'annotation': 'number of examples with graph match with strings anonymized'}
    )
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
    def precision(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum(self.precision_list) / self.num_total_examples

    @property
    def recall(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum(self.recall_list) / self.num_total_examples

    @property
    def f1(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum(self.f1_list) / self.num_total_examples

    @property
    def precision_anonym_string(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum(self.precision_anonym_string_list) / self.num_total_examples

    @property
    def recall_anonym_string(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum(self.recall_anonym_string_list) / self.num_total_examples

    @property
    def f1_anonym_string(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return sum(self.f1_anonym_string_list) / self.num_total_examples

    def add(self, ref_graph: Graph, pred_graph: Graph):

        if ref_graph.is_identical(pred_graph):
            self.num_matched_graphs += 1

        # test graph match with strings all anonymized (to see quality of string generation or copying)
        if anonymize_strings_in_graph(ref_graph).is_identical(anonymize_strings_in_graph(pred_graph)):
            self.num_matched_graphs_anonym_string += 1

        self.num_total_examples += 1

        # tuple match scores
        precision, recall, f1 = graph_tuple_match(ref_graph, pred_graph)

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
        precision_anonym, recall_anonym, f1_anonym = \
            graph_tuple_match(anonymize_strings_in_graph(ref_graph),
                              anonymize_strings_in_graph(pred_graph)
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
        # graph tuple match
        output += f'tuple_match_average_precision: {self.precision:.5f}' + '\n'
        output += f'tuple_match_average_recall: {self.recall:.5f}' + '\n'
        output += f'tuple_match_average_f1: {self.f1:.5f}' + '\n'
        # graph tuple match with anonymized strings
        output += f'tuple_match_strings_anonymized_average_precision: {self.precision_anonym_string:.5f}' + '\n'
        output += f'tuple_match_strings_anonymized_average_recall: {self.recall_anonym_string:.5f}' + '\n'
        output += f'tuple_match_strings_anonymized_average_f1: {self.f1_anonym_string:.5f}'

        return output


def evaluate_graph_tuples(
    test_actions_file: str,
    gold_actions_file: str,
    ) -> GraphTupleMatchScores:
    """Evaluate generated graph (represented by actions) with tuple match metric
    (including precision, recall, and F1).

    Args:
        test_actions_file (str): [description]
        gold_actions_file (str, optional): [description]. Defaults to None.

    Returns:
        GraphTupleMatchScores: [description]
    """
    test_actions = read_string_sentences(test_actions_file)
    gold_actions = read_string_sentences(gold_actions_file)

    graph_tuple_match_scores = GraphTupleMatchScores()

    for idx, (gold_act, test_act) in tqdm(enumerate(zip(gold_actions, test_actions)),
                                          unit=' graphs (action sequences)', desc='Evaluation - Graph Tuple Match'):
        gold_graph = actions_to_graph(gold_act.split(' '))
        test_graph = actions_to_graph(test_act.split(' '))

        graph_tuple_match_scores.add(
            ref_graph=gold_graph,
            pred_graph=test_graph
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    graph_tuple_match_scores = evaluate_graph_tuples(
        test_actions_file=args.test_actions,
        gold_actions_file=args.gold_actions,
        )

    with open(args.out_file, 'w') as f:
        print(graph_tuple_match_scores, file=f)
        print(f'Evaluation - Graph Tuple Match - results saved to {args.out_file}')
