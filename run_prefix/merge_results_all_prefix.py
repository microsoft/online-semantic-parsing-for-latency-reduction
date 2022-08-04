# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Merge all beam decoded results for a validation set for different prefix steps."""
import argparse
from itertools import chain
import json
import os
from typing import List, Dict

from tqdm import tqdm

from calflow_parsing.io import read_string_sentences


subset = 'valid'
max_len = 60
data_sets = [f'{subset}-prefix{x}t' for x in range(max_len + 1)]

beam_size = 1
# beam_size = 5
model = 'checkpoint_last'

rootdir = '../SAVE'

# order-top
model_folder = ('exp_prefix-last-8ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
                '/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2')
# model_folder = ('exp_prefix-last-10ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
#                 '/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2')
# model_folder = ('exp_prefix-last-5ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
#                 '/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2')
# model_folder = ('exp_prefix-11ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
#                 '/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2')
model_folder = ('exp_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
                '/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2')

# order-bot
# model_folder = ('exp_prefix-last-8ps_order-bot_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
#                 '/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2')
# model_folder = ('exp_prefix-last-10ps_order-bot_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
#                 '/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2')
# model_folder = ('exp_prefix-last-5ps_order-bot_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
#                 '/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2')
# model_folder = ('exp_prefix-11ps_order-bot_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
#                 '/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2')
# model_folder = ('exp_prefix-p100_order-bot_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
#                 '/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2')

data_folder = '../DATA/processed/smcalflow2.0/act-top_src-ct1-npwa_tgt-cp-str_abs-prefix-all/oracle'


def parse_args():
    parser = argparse.ArgumentParser(description='merge all beam decoding results with every prefix length data validation')
    parser.add_argument('--rootdir', type=str, default=rootdir,
                        help='Root directory of experiment folders')
    parser.add_argument('--model_folder', type=str, default=model_folder,
                        help='Experiment folder prefix to be matched')
    parser.add_argument('--data_folder', type=str, default=data_folder,
                        help='Data folder that stores the prefix source files')
    parser.add_argument('--subset', type=str, default=subset,
                        help='test data subset')
    parser.add_argument('--max_len', type=int, default=max_len,
                        help='max length in the test data')
    # parser.add_argument('--data_sets', type=str, nargs='*', default=data_sets,
    #                     help='data sets to collect scores')
    parser.add_argument('--beam_size', type=int, default=beam_size,
                        help='beam size in the decoding')
    parser.add_argument('--model', type=str, default=model,
                        help='model checkpoint names to collect scores')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the results')
    args = parser.parse_args()

    return args


def extract_prefix_linenos_in_files(data_sets, data_folder):
    """Extract line numbers of each utterance with different prefix lengths in a group of files."""
    # read in all line numbers in the original data
    prefix_idxs_all: List[List[int]] = []
    for i, d in tqdm(enumerate(data_sets), unit=' idxs_files'):
        idxs_file = os.path.join(data_folder, f'{d}.idxs')
        idxs = read_string_sentences(idxs_file)
        # convert from str to int
        idxs = list(map(int, idxs))

        prefix_idxs_all.append(idxs)

    # extract which line to index in each file for each prefix file
    all_linenos = sorted(list(set(chain.from_iterable(prefix_idxs_all))))
    assert min(all_linenos) == 0 and max(all_linenos) == len(all_linenos) - 1, 'line numbers are not consecutive'
    index_in_different_files: List[List[int]] = []
    for lineno in all_linenos:
        index_in_different_files_current = []
        for idxs in prefix_idxs_all:
            if lineno in idxs:
                index_in_different_files_current.append(idxs.index(lineno))
            else:
                break
        index_in_different_files.append(index_in_different_files_current)

    return index_in_different_files, prefix_idxs_all


def merge_prefix_results(beam_dir, data_sets, data_folder, model='checkpoint_last'):
    """Merge beam results with all prefix length data.

    NOTE we assume the `data_sets` contains consecutive prefix lengths, and it has to start from 0 length.
    """
    prefix_results: List[Dict[int, Dict]] = []
    for d in tqdm(data_sets, unit=' decoded_hypo_files'):
        ps = json.load(open(os.path.join(beam_dir, f'{d}_{model}.hypos.json')))
        prefix_results.append(ps)

    index_in_different_files, prefix_idxs_all = extract_prefix_linenos_in_files(data_sets, data_folder)
    print('index in different prefix files collected!')

    all_prefix_predictions = {}
    for i, idx_files in tqdm(enumerate(index_in_different_files), unit=' examples'):
        prefix_predictions = {}
        num_prefixes = len(idx_files)
        # breakpoint()
        try:
            for j, idx in enumerate(idx_files):
                prefix_predictions[j] = prefix_results[j][str(idx)]    # List[Dict], each one is for one beam
        except:
            breakpoint()

        all_prefix_predictions.setdefault(i, {})['predictions'] = prefix_predictions
        # below `j` is the last prefix index, which is the case of full utterance
        all_prefix_predictions[i]['src_str'] = prefix_predictions[j][0]['src_str']
        assert '<mask>' not in all_prefix_predictions[i]['src_str'], 'seems not full utterance here'
        all_prefix_predictions[i]['reference'] = prefix_predictions[j][0]['reference']
        # extract context and utterance
        context = all_prefix_predictions[i]['src_str'].split('__User')[:-1]
        utterance = all_prefix_predictions[i]['src_str'].split('__User')[-1]
        utterance = utterance.split()[:-1]
        len_utter = len(utterance)
        all_prefix_predictions[i]['context'] = '__User'.join(context)
        all_prefix_predictions[i]['utterance'] = ' '.join(utterance)
        all_prefix_predictions[i]['len_utter'] = len_utter
        all_prefix_predictions[i]['len_src'] = len(all_prefix_predictions[i]['src_str'].split())

    # `all_prefix_predictions` is a dictionary containing outputs for all sentences in the test dataset
    # keys and values:
    # data example id: {
    #   'predictions': {
    #       0: List[
    #           {
    #           'actions': List[str],
    #           'step_scores': List[float],
    #           ...
    #            }
    #           ]        # for different beams
    #       1: List [
    #           {
    #           'actions': List[str],
    #           'step_scores': List[float],
    #           ...
    #           }
    #           ]        # for different beams
    #       # for different prefix length
    #       }
    #   'src_str': str,
    #   'context': str,
    #   'utterance': str,
    #   'len_utter': int,
    #   ...
    #   }

    return all_prefix_predictions


if __name__ == '__main__':
    args = parse_args()

    data_sets = [f'{args.subset}-prefix{x}t' for x in range(args.max_len + 1)]
    # data_sets = args.data_sets

    data_folder = args.data_folder

    model_folder = os.path.join(args.rootdir, args.model_folder)

    beam_dir = os.path.join(model_folder, f'beam{args.beam_size}')

    save_path = os.path.join(beam_dir, f'{args.subset}-prefixallt.hypos.json')

    all_prefix_predictions = merge_prefix_results(
        beam_dir=beam_dir,
        data_sets=data_sets,
        data_folder=data_folder,
        model=args.model,
        )

    print('Saving results ...')
    json.dump(all_prefix_predictions, open(save_path, 'w'))
    print(f'All {args.subset} prefix prediction results with beam size {args.beam_size} saved to: {save_path}')
