# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
from copy import deepcopy
from typing import Callable, List, Tuple
from dataclasses import dataclass, field

from tqdm import tqdm

from dataflow.core.lispress import parse_lispress, lispress_to_program, program_to_lispress, render_compact
from dataflow.core.linearize import seq_to_lispress
from calflow_parsing.graph import Graph
from calflow_parsing.calflow_graph import actions_to_graph, ProgramGraph
from calflow_parsing.io import read_string_sentences
from calflow_parsing.calflow_graph_lispress import _CONVERSION_ERROR


STRING_ANONYM = 'STRING'


def graph_match_from_actions(gold_actions: List[str], test_actions: List[str]) -> Tuple[bool, Graph, Graph]:
    graph1 = actions_to_graph(gold_actions)
    graph2 = actions_to_graph(test_actions)
    matched = graph1.is_identical(graph2)
    return matched, graph1, graph2


def to_canonical_form(lispress: str, tokenized=False) -> str:
    """Returns canonical form of a lispress.

    The canonical form is un-tokenized and compact; it also sorts named arguments in alphabetical order.
    """
    if tokenized:
        lispress = seq_to_lispress(lispress.split(" "))
    else:
        lispress = parse_lispress(lispress)
    program, _ = lispress_to_program(lispress, 0)
    round_tripped = program_to_lispress(program)
    return render_compact(round_tripped)


def lispress_match(gold_lispress: str, test_lispress: str, tokenized: bool = False) -> bool:
    """See if two lispress (untokenized) strings are exact matches."""
    if test_lispress == gold_lispress:
        matched = True
    else:
        # convert to canonical form
        gold_canonical = to_canonical_form(gold_lispress, tokenized=tokenized)
        test_canonical = to_canonical_form(test_lispress, tokenized=tokenized)
        if gold_canonical == test_canonical:
            matched = True
        else:
            matched = False

    return matched


def graph_match_from_action_files(gold_actions_file: str, test_actions_file: str):
    with open(gold_actions_file, 'r') as gold_actions, open(test_actions_file, 'r') as test_actions:
        total_num = 0
        match_num = 0
        for actions1, actions2 in tqdm(zip(gold_actions, test_actions)):
            if not actions1.strip():
                continue

            actions1 = actions1.strip().split()
            actions2 = actions2.strip().split()

            matched, graph1, graph2 = graph_match_from_actions(actions1, actions2)
            if matched:
                match_num += 1

            # if not matched:
            #     print('graphs not matching:')
            #     print('-' * 20 + ' graph1 ' + '-' * 20)
            #     print(graph1)
            #     print('-' * 20 + ' graph2 ' + '-' * 20)
            #     print(graph2)
            #     breakpoint()

            total_num += 1

        em = match_num / total_num

    return em, total_num, match_num


def exact_match_from_files(gold_file: str, test_file: str):
    with open(gold_file, 'r') as f, open(test_file, 'r') as g:
        total_num = 0
        match_num = 0
        for line1, line2 in zip(f, g):
            if not line1.strip():
                continue

            matched = line1.strip() == line2.strip()
            if matched:
                match_num += 1

            total_num += 1

        em = match_num / total_num

    return em, total_num, match_num


def lispress_match_from_files(gold_lispress_file: str, test_lispress_file: str, tokenized: bool = False):
    with open(gold_lispress_file, 'r') as gold_lispress, open(test_lispress_file, 'r') as test_lispress:
        total_num = 0
        match_num = 0
        for lispress1, lispress2 in tqdm(zip(gold_lispress, test_lispress)):
            if not lispress1.strip():
                continue

            lispress1 = lispress1.strip()
            lispress2 = lispress2.strip()

            matched = lispress_match(lispress1, lispress2, tokenized=tokenized)

            if matched:
                match_num += 1

            total_num += 1

        em = match_num / total_num

    return em, total_num, match_num


@dataclass
class ExactMatchScores:
    num_total_examples: int = field(default=0, metadata={'annotation': 'total number of examples evaluated'})
    num_matched_graphs: int = field(default=0, metadata={'annotation': 'number of examples with graph match'})
    num_matched_graphs_detok_string: int = field(
        default=0,
        metadata={'annotation': 'number of examples with graph match with strings detokenized'}
    )
    num_matched_lispress: int = field(default=0, metadata={'annotation': 'number of examples with lispress match'})
    num_matched_graph_only: int = field(
        default=0,
        metadata={'annotation': 'number of examples with graph match but not lispress match'}
        )
    num_matched_graphs_anonym_string: int = field(
        default=0,
        metadata={'annotation': 'number of examples with graph match with strings anonymized'}
    )
    num_errors_graph_to_lispress: int = field(
        default=0,
        metadata={'annotation':
            'number of examples where an error occurred during the conversion from graph to lispress'}
        )
    dataset: str = field(default='smcalflow',
                         metadata={'annotation': 'dataset source; choose from ["smcalflow", "treedst"]'})

    @property
    def accuracy(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return self.num_matched_graphs / self.num_total_examples

    @property
    def accuracy_lispress(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return self.num_matched_lispress / self.num_total_examples

    @property
    def accuracy_only_graph_not_lispress(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return self.num_matched_graph_only / self.num_total_examples

    @property
    def accuracy_graph_strings_detokenized(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return self.num_matched_graphs_detok_string / self.num_total_examples

    @property
    def accuracy_graph_strings_anonymized(self) -> float:
        if self.num_total_examples == 0:
            return 0
        return self.num_matched_graphs_anonym_string / self.num_total_examples

    def add(self, ref_lispress: str, pred_lispress: str, ref_graph: Graph, pred_graph: Graph,
            ref_graph_detok_string: Graph = None):
        if ref_lispress == pred_lispress:
            self.num_matched_lispress += 1
            matched_lispress = True
        else:
            matched_lispress = False

        if ref_graph.is_identical(pred_graph):
            self.num_matched_graphs += 1
            matched_graph = True
        else:
            matched_graph = False

        if matched_graph and not matched_lispress:
            self.num_matched_graph_only += 1

        if not matched_graph and matched_lispress:
            # raise ValueError('can not have graph unmatched while lispress matched')
            import warnings
            warnings.warn('we have graph unmatched while lispress matched')
            print('[gold graph]:')
            print(ref_graph)
            print('[predicted graph]:')
            print(pred_graph)
            print('[gold and predicted lispress]:')
            print(ref_lispress)

        # detokenize the strings in the graph nodes and measure graph exact match
        # NOTE for this the gold graph should be directly derived from the original lispress -> program -> graph,
        #      where no tokenization was done
        if ref_graph_detok_string is not None:
            pred_graph_detok_string = ProgramGraph(dataset=self.dataset).graph_detokenize_string(pred_graph)
            if ref_graph_detok_string.is_identical(pred_graph_detok_string):
                self.num_matched_graphs_detok_string += 1

        # test graph match with strings all anonymized (to see quality of string generation or copying)
        if anonymize_strings_in_graph(ref_graph).is_identical(anonymize_strings_in_graph(pred_graph)):
            self.num_matched_graphs_anonym_string += 1

        if pred_lispress == _CONVERSION_ERROR:
            self.num_errors_graph_to_lispress += 1

        self.num_total_examples += 1

    def __str__(self):
        output = ''
        output += f'num_total_examples: {self.num_total_examples}' + '\n'
        output += f'num_matched_graphs: {self.num_matched_graphs}' + '\n'
        output += f'num_matched_lispress: {self.num_matched_lispress}' + '\n'
        output += f'num_matched_graph_only: {self.num_matched_graph_only}' + '\n'
        output += f'num_matched_graphs_detok_string: {self.num_matched_graphs_detok_string}' + '\n'
        output += f'num_matched_graphs_anonym_string: {self.num_matched_graphs_anonym_string}' + '\n'
        output += f'num_errors_graph_to_lispress: {self.num_errors_graph_to_lispress}' + '\n'
        output += f'exact_match_graphs: {self.accuracy:.5f}' + '\n'
        output += f'exact_match_graph_strings_detokenized: {self.accuracy_graph_strings_detokenized:.5f}' + '\n'
        output += f'exact_match_lispress: {self.accuracy_lispress:.5f}' + '\n'
        output += f'exact_match_graph_only_not_lispress: {self.accuracy_only_graph_not_lispress:.5f}' + '\n'
        output += f'exact_match_graph_strings_anonymized: {self.accuracy_graph_strings_anonymized:.5f}'

        return output


def anonymize_strings_in_graph(graph: Graph) -> Graph:
    """Annoymize the string nodes with names marked by quotes in the graph nodes. The resulted graph can then be used
    to measure the matching accuracy, to indirectly see the quality of the string generations (or copies).

    Args:
        graph (Graph): [description]

    Returns:
        Graph: [description]
    """
    graph_string_anonym = deepcopy(graph)
    for nid, name in graph_string_anonym.nodes.items():
        if name.startswith('"') and name.endswith('"'):
            graph_string_anonym.nodes[nid] = f'"{STRING_ANONYM}"'
    return graph_string_anonym


def evaluate_graph_and_lispress(
    gold_lispress_file: str,
    test_actions_file: str,
    gold_actions_file: str = None,
    test_lispress_file: str = None,
    dataset: str = 'smcalflow',
    ) -> ExactMatchScores:
    """Evaluate generated program (graphs, lispress, represented by actions) with exact match metric.

    Args:
        gold_lispress_file (str): [description]
        test_actions_file (str): [description]
        gold_actions_file (str, optional): [description]. Defaults to None.
        test_lispress_file (str, optional): [description]. Defaults to None.

    Returns:
        ExactMatchScores: [description]
    """
    gold_lispress = read_string_sentences(gold_lispress_file)
    test_actions = read_string_sentences(test_actions_file)
    gold_actions = read_string_sentences(gold_actions_file) if gold_actions_file is not None else None
    test_lispress = read_string_sentences(test_lispress_file) if test_lispress_file is not None else None

    program_graph_methods = ProgramGraph(type_args_coversion='atomic', dataset=dataset)

    exact_match_scores = ExactMatchScores(dataset=dataset)

    for idx, (gold_lisp, test_act) in tqdm(enumerate(zip(gold_lispress, test_actions)),
                                           unit=' lispress', desc='Evaluation - Exact Match'):

        # if gold_actions is not None:
        #     gold_graph = actions_to_graph(gold_actions[idx].split(' '))
        # else:
        #     gold_graph = program_graph_methods.lispress_to_graph(gold_lisp)

        gold_graph_detok_string = program_graph_methods.lispress_to_graph(gold_lisp)
        # NOTE here gold_graph corresponds to the actions (with all string tokenization)
        if gold_actions is not None:
            gold_graph = actions_to_graph(gold_actions[idx].split(' '))
        else:
            gold_graph = program_graph_methods.graph_tokenize_string(gold_graph_detok_string)

        test_graph = actions_to_graph(test_act.split(' '))

        if test_lispress is not None:
            test_lisp = test_lispress[idx]
        else:
            try:
                test_lisp = program_graph_methods.graph_to_lispress(test_graph)
            except:
                test_lisp = _CONVERSION_ERROR

        exact_match_scores.add(ref_lispress=gold_lisp,
                               pred_lispress=test_lisp,
                               ref_graph=gold_graph,
                               pred_graph=test_graph,
                               ref_graph_detok_string=gold_graph_detok_string)

    return exact_match_scores


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of Exact Match of graph and lispress')
    parser.add_argument('--gold_lispress', type=str,
                        help='input file with the gold lispress strings (untokenized)')
    parser.add_argument('--test_actions', type=str,
                        help='input file with the testing graph actions')
    parser.add_argument('--gold_actions', type=str,
                        help='input file with the gold graph actions')
    parser.add_argument('--test_lispress', type=str,
                        help='input file with the testing lispress strings (untokenized)')
    parser.add_argument('--out_file', type=str,
                        help='output file to store the resulted exact match scores')
    parser.add_argument('--dataset', type=str, default='smcalflow', choices=['smcalflow', 'treedst'],
                        help='dataset that we are dealing with')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    exact_match_scores = evaluate_graph_and_lispress(
        gold_lispress_file=args.gold_lispress,
        test_actions_file=args.test_actions,
        gold_actions_file=args.gold_actions,
        test_lispress_file=args.test_lispress,
        dataset=args.dataset,
        )

    with open(args.out_file, 'w') as f:
        print(exact_match_scores, file=f)
        print(f'Evaluation - Exact Match - results saved to {args.out_file}')
