# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Collect all the results from different exps and different models (with varying hyper-parameter setups).
"""
import argparse
import os
from typing import List, Tuple
import json
import sys
sys.path.insert(0, './run')    # NOTE we call the script from the root dir (so not adding '../run')
sys.path.insert(0, '../run')    # NOTE for running the script inside some folder, e.g. jupyter notebook TODO clean this
from collect_scores_all import collect_final_scores

from tqdm import tqdm

from calflow_parsing.io import mkdir_for_file


rootdir = '../SAVE'
data_sets = ['valid-prefix100p', 'valid-prefix90p', 'valid-prefix80p', 'valid-prefix70p', 'valid-prefix60p',
             'valid-prefix50p', 'valid-prefix40p', 'valid-prefix30p', 'valid-prefix20p', 'valid-prefix10p',
             'valid-prefix0p']
models = ['last']
# score_names = ['em', 'tm', 'mod-graph.em', 'mod-graph.tm']
# score_names = ['apim', 'mod-graph.apim']
score_names = ['em', 'tm', 'mod-graph.em', 'mod-graph.tm', 'apim', 'mod-graph.apim']
score_items = []


def yellow(string):
    return "\033[93m%s\033[0m" % string


def red(string):
    return "\033[91m%s\033[0m" % string


def parse_args():
    parser = argparse.ArgumentParser(description='collect all exp results with prefix percentage data validation')
    parser.add_argument('--rootdir', type=str, default=rootdir,
                        help='Root directory of experiment folders')
    parser.add_argument('--exp_prefix', type=str, default='exp_prefix',
                        help='Experiment folder prefix to be matched')
    parser.add_argument('--data_sets', type=str, nargs='*', default=data_sets,
                        help='data sets to collect scores')
    parser.add_argument('--models', type=str, nargs='*', default=models,
                        help='model checkpoint names to collect scores')
    parser.add_argument('--score_names', type=str, nargs='*', default=score_names,
                        help='postfix of the score files')
    parser.add_argument('--score_items', type=str, nargs='*', default=score_items,
                        help='item of the scores in the score file, when there is more than one (e.g. exact match for calflow data)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the results')
    args = parser.parse_args()

    return args


def collect_results_all_exps(rootdir, exp_prefix, data_sets, models, score_names, score_items=None):
    all_exp_results = {}
    for exp_path in tqdm(os.listdir(rootdir), desc='scanning exp dirs'):
        if os.path.isdir(os.path.join(rootdir, exp_path)) and exp_path.startswith(exp_prefix):
            print(red(f'[{exp_path}]'))

            for model_path in os.listdir(os.path.join(rootdir, exp_path)):
                if os.path.isdir(os.path.join(rootdir, exp_path, model_path)) and model_path.startswith('models_'):
                    print('-' * 10, model_path)

                    beam_results = collect_final_scores(
                        os.path.join(rootdir, exp_path, model_path),
                        data_sets,
                        models,
                        score_names,
                        score_items
                        )
                    all_exp_results.setdefault(exp_path, {})[model_path] = beam_results

    # beam_results is a dictionary, with key levels:
    # beam_size -> model_name -> data_set -> score_name -> score_item_name -> score_item_value

    # all_exp_results is a dictionary, with 2 more key levels:
    # exp_path_name (not full path) -> model_path_name (not full path) -> beam_results

    return all_exp_results


def extract_prefix_score_list(
    beam_results,
    data_sets: List[str] = data_sets,
    score_name_and_items: List[Tuple[str, str]] = [('em', 'exact_match_graphs')],
    beam_size: int = 1,
    model: str = 'last'
    ):
    prefix_scores = {}
    scores = beam_results[str(beam_size)][model]
    for score_name, score_item in score_name_and_items:
        for ds in data_sets:
            prefix_scores.setdefault(score_item, []).append(scores[ds][score_name][score_item])

    # prefix_scores is a dictionary, with key and item:
    # score_item_name: list of scores corresponding to the prefix data
    return prefix_scores


if __name__ == '__main__':
    args = parse_args()

    all_exp_results = collect_results_all_exps(
        args.rootdir,
        args.exp_prefix,
        args.data_sets,
        args.models,
        args.score_names,
        args.score_items
        )

    cat_score_names = '_'.join(map(lambda x: x.replace('.', '-'), args.score_names))
    save_path = os.path.join('results_collection', f'collected_all_{cat_score_names}_scores_from_{args.exp_prefix}.json')

    # breakpoint()

    mkdir_for_file(save_path)

    json.dump(all_exp_results, open(save_path, 'w'))
    print(f'Collected all scores saved at: {save_path}')
