# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Based on the graph actions, recover the underlying graph, and covert back to lispress expressions.
This is the revert process of data conversion from raw lispress data to graph actions.
"""
import argparse
from typing import List, Tuple

from tqdm import tqdm

from .graph import Graph
from .calflow_graph import ProgramGraph, actions_to_graph
from .io import write_string_sentences


_PARSE_ERROR_LISPRESS = '(parseError #(InvalidLispress "")'
_CONVERSION_ERROR = '(ConversionErrorGraphToLispress)'


def play(in_source: str = None,
         in_actions: str = None,
         out_lispress: str = None,
         dataset: str = 'smcalflow',
         ) -> Tuple[List[Graph], List[str]]:
    """Play the graph actions to recover the underlying graph and back to the lispress as well.

    Args:
        in_source (str, optional): [description]. Defaults to None.
        in_actions (str, optional): [description]. Defaults to None.
        out_lispress (str, optional): [description]. Defaults to None.

    Returns:
        Tuple[List[Graph], List[str]]: [description]
    """
    # NOTE in_source is currently not used; would be useful with e.g. alignments or copying.
    graphs = []
    lisps_tokstr = []
    lisps = []

    program_graph_methods = ProgramGraph(type_args_coversion='atomic', dataset=dataset)

    with open(in_actions, 'r') as f:
        for actions in tqdm(f, unit=' actions', desc='Conversion to lispress'):
            if not actions.strip():
                continue

            actions = actions.strip().split(' ')

            # play the actions and return graph
            graph = actions_to_graph(actions)

            graphs.append(graph)

            # return the lispress with tokenized strings in graph
            try:
                lispress_tokstr = program_graph_methods.graph_to_lispress(graph)
            except:
                lispress_tokstr = _CONVERSION_ERROR
            lisps_tokstr.append(lispress_tokstr)

            # detokenize the string nodes in graph
            graph = program_graph_methods.graph_detokenize_string(graph)

            # return the lispress
            try:
                lispress = program_graph_methods.graph_to_lispress(graph)
            except:
                lispress = _CONVERSION_ERROR
            lisps.append(lispress)

    out_lispress_tokstr = out_lispress + '.tokstr'
    write_string_sentences(out_lispress_tokstr, lisps_tokstr)
    write_string_sentences(out_lispress, lisps)

    return graphs, lisps


def parse_args():
    parser = argparse.ArgumentParser(description='Recover graph and lispress expressions from actions')
    parser.add_argument('--in_source', type=str,
                        help='input file with the processed source tokens (including the turn contexts)')
    parser.add_argument('--in_actions', type=str,
                        help='input file with the graph actions')
    parser.add_argument('--out_lispress', type=str,
                        help='recovered lispress from the graph actions')
    parser.add_argument('--dataset', type=str, default='smcalflow', choices=['smcalflow', 'treedst'],
                        help='dataset that we are dealing with')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    play(in_source=args.in_source,
         in_actions=args.in_actions,
         out_lispress=args.out_lispress,
         dataset=args.dataset)
