# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Collect all scores presented in the decoding result folder,
where one can specify multiple score file name suffix, and each score file can contain
multiple score items (to be converted to a dictionary).
"""

import os
import re
import argparse
from typing import List
import json


# beam results dir name reges
beam_dir_re = re.compile(r'beam([0-9]+)')

# results file name regex
results_re = re.compile(r'(.+)_checkpoint_(.+?)\.(.+)')  # +? for non-greedy matching; only match the first "\."
# smatch_re = re.compile(r'valid_checkpoint([0-9]+)\.smatch')
# smatch_re_wiki = re.compile(r'valid_heckpoint([0-9]+)\.wiki\.smatch')

# model names to consider
# models = ['last', 'wiki-smatch_best1', 'wiki-smatch_top3-avg', 'wiki-smatch_top5-avg']
models = ['last']

# results file content regex: to extract (key, value) pairs from the score file
item_results_re = re.compile(r'([^:\s]+):\s*([0-9\.]+)')


def yellow(string):
    return "\033[93m%s\033[0m" % string


def red(string):
    return "\033[91m%s\033[0m" % string


def parse_args():
    parser = argparse.ArgumentParser(description='Collect model results')
    parser.add_argument('checkpoints', type=str, default='../SAVE/exp_debug/models_ep120_seed42',
                        help='folder containing saved model checkpoints for a single training')
    parser.add_argument('--data_sets', type=str, nargs='*', default=['valid', 'test'],
                        help='data sets to collect scores')
    parser.add_argument('--models', type=str, nargs='*', default=models,
                        help='model checkpoint names to collect scores')
    parser.add_argument('--score_names', type=str, nargs='*', default=['wiki.smatch'],
                        help='postfix of the score files')
    parser.add_argument('--score_items', type=str, nargs='*', default=None,
                        help='item of the scores in the score file, when there is more than one (e.g. exact match for calflow data)')
    parser.add_argument('--ndigits', type=int, default=2,
                        help='number of digits after the decimal point')
    # parser.add_argument('--save_name', type=str, default='collected_wiki-smatch_scores.txt',
    #                     help='save name for the collection of results')
    args = parser.parse_args()
    return args


def get_score_dict_from_log(file_path, score_items: List[str] = None):
    """Read the file containing the scores, and convert it to a dictionary.
    If `score_items` is not `None`, we only extract the score items belonging to the specified list.
    """
    results = {}
    with open(file_path, 'r') as fid:
        for line in fid:
            scores_all = item_results_re.findall(line)    # list of tuples, e.g. [('exact_match', 0.888), ...]
            for key, value in scores_all:
                if (score_items and score_items != ['']) and key not in score_items:
                    # NOTE when input "" from bash script `score_items` would be ['']
                    continue
                results[key] = float(value)

    return results


def get_scores_from_beam_dir(beam_dir, data_sets, models, score_names, score_items=None):
    # get results from a beam directory with a single beam size
    results_dict = {}

    for dfile in os.listdir(beam_dir):

        if not results_re.match(dfile):
            continue

        data_set, model_name, sname = results_re.match(dfile).groups()

        if (data_set in data_sets) and (model_name in models) and (sname in score_names):
            score_dict = get_score_dict_from_log(os.path.join(beam_dir, dfile), score_items)
            results_dict.setdefault(model_name, {}).setdefault(data_set, {})[sname] = score_dict    # could be None

    return results_dict


def collect_final_scores(checkpoint_folder, data_sets, models, score_names, score_items=None):
    beam_sizes = []
    for name in os.listdir(checkpoint_folder):

        if not beam_dir_re.match(name):
            continue

        beam_size, = beam_dir_re.match(name).groups()
        beam_sizes.append(int(beam_size))

    beam_sizes = sorted(beam_sizes)
    beam_results = {}

    for bs in beam_sizes:

        beam_dir = os.path.join(checkpoint_folder, f'beam{bs}')

        results_dict = get_scores_from_beam_dir(beam_dir, data_sets, models, score_names, score_items)
        beam_results[bs] = results_dict

    # beam_results is a dictionary, with key levels:
    # beam_size -> model_name -> data_set -> score_name -> score_item_name -> score_item_value
    return beam_results


if __name__ == '__main__':
    args = parse_args()

    # TODO update this piece
    # if 'wiki.smatch' not in args.score_names:
    #     args.models = [m.replace('wiki-smatch', args.score_name.replace('.', '-')) for m in models]

    beam_results = collect_final_scores(args.checkpoints, args.data_sets, args.models,
                                        args.score_names, args.score_items)

    # print(beam_results[1]['last'])
    # breakpoint()

    cat_score_names = '_'.join(map(lambda x: x.replace('.', '-'), args.score_names))
    save_path = os.path.join(args.checkpoints, f'collected_all_{cat_score_names}_scores.json')

    json.dump(beam_results, open(save_path, 'w'))
    print(f'Collected all scores saved at: {save_path}')
