# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Merge results for a validation set for different prefix steps, obtained from the pipeline
de-noising language model completion (DLM) completion (e.g. beam 5) -> full parser (e.g. beam 1) -> target graphs.

Results for different prefix lengths are concatanated in a single file, both for the DLM decoding results (utterance
completion) and the parser decoding results based on that (graph generation).

The merged results will include both scores for the DLM utterance completion and scores for the parser action
generation (scores: log likelihood; might include one extra step at the end for </s> eos token; might not be of
the same length as the finished utterance since the BART generation is based on BPE).
"""

import argparse
from itertools import chain
import json
import math
import os
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from calflow_parsing.io import read_string_sentences


rootdir = '/n/tata_ddos_ceph/jzhou/incremental-interpretation'

data_folder = 'DATA/processed/smcalflow2.0/src-ct1-npwa_utter-dlm_abs-prefix-all/oracle'

subset = 'valid'
max_len = 60
data_sets = [f'{subset}-prefix{x}t' for x in range(max_len + 1)]

only_dlm = 0
# language completion model
dlm_beam_size = 5
# dlm_beam_size = 1
dlm_model = 'checkpoint_best'

dlm_model_folder = (
    'SAVE/'
    'exp_utter_abs-prefix-all_dlm_bart-large/'
    'models_ep12_seed42_fp16-lr5e-05-schedulerpolynomial_decay-mt2048x4-wm500-dp0.1-cn0.1'
    )

# parsing model
# order-top
model_folder = (
    'SAVE/'
    'exp_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/'
    'models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2'
    )

beam_size = 1
model = 'checkpoint_last'


# ========== for TreeDST ==========

rootdir = '/n/tata_ddos_ceph/jzhou/incremental-interpretation'

data_folder = 'DATA/processed/treedst/src-ct1-npwa_utter-dlm_abs-prefix-all/oracle'

subset = 'valid'
max_len = 43
data_sets = [f'{subset}-prefix{x}t' for x in range(max_len + 1)]

only_dlm = 0
# language completion model
# dlm_beam_size = 5
dlm_beam_size = 1
dlm_model = 'checkpoint_best'

dlm_model_folder = (
    'SAVE/'
    'exp_treedst_utter_abs-prefix-all_dlm_bart-large/'
    'models_ep12_seed42_fp16-lr5e-05-schedulerpolynomial_decay-mt2048x4-wm500-dp0.1-cn0.1'
)

# parsing model
# order-top
model_folder = (
    'SAVE/'
    'exp_treedst_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/'
    'models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2'
)

beam_size = 1
model = 'checkpoint_last'

# ==============================


def extract_prefix_linenos_in_file(data_sets, data_folder):
    """Extract line numbers of each utterance with incremental prefix lengths in a single concatenated file,
    where it is a stack of prefixes of different absolute length, e.g. all prefixes with 0 tokens + all prefixes with
    1 tokens + ...

    Args:
        data_sets (List[str]): individual absolute length prefix files.
        data_folder (str): path of the oracle folder that stores the individual absolute prefix length file and the
            indices of the original data that it includes.

    Returns:
        index_in_stacked_file (List[List[int]]): incremental prefix line numbers for each utterance in the single file.
        prefix_idxs_all (List[List[int]]): the original data index included for each absolute prefix length.
    """
    # read in all line numbers in the original data
    prefix_idxs_all: List[List[int]] = []
    for i, d in tqdm(enumerate(data_sets), unit=' idxs_files'):
        idxs_file = os.path.join(data_folder, f'{d}.idxs')
        idxs = read_string_sentences(idxs_file)
        # convert from str to int
        idxs = list(map(int, idxs))

        prefix_idxs_all.append(idxs)

    # extract the list of line numbers for incremental prefixes for each utterance in the concatenated file
    all_linenos = sorted(list(set(chain.from_iterable(prefix_idxs_all))))
    assert min(all_linenos) == 0 and max(all_linenos) == len(all_linenos) - 1, 'line numbers are not consecutive'
    index_in_stacked_file: List[List[int]] = []
    stacked_indices = np.array(list(chain.from_iterable(prefix_idxs_all)))
    num_passed_lines = 0
    for lineno in all_linenos:
        index_in_stacked_file_current = (stacked_indices == lineno).nonzero()[0].tolist()
        index_in_stacked_file.append(index_in_stacked_file_current)
        num_passed_lines += len(index_in_stacked_file_current)

    assert num_passed_lines == len(stacked_indices)

    return index_in_stacked_file, prefix_idxs_all


def merge_prefix_dlm_to_parser_results(*,
                                       beam_dir,
                                       dlm_beam_dir,
                                       dlm_beam_size,
                                       data_sets,
                                       data_folder,
                                       only_dlm=False,
                                       data_subset='valid',
                                       model='checkpoint_last',
                                       dlm_model='checkpoint_best'):
    """Merge beam results with all prefix data, which is passed by dlm completion to full parser.
    The collected results contain scores for both beam procedures.

    NOTE
    - In `dlm_beam_dir`, there are `dlm_beam_size` number of separate sets of beam results, stored in individual files.
    - In `beam_dir`, there are `dlm_beam_size` number of parser beam decoding results corresponding to above, but with
      different beam results stored in a single file "*.hypos.json".

    Args:
        beam_dir (str): beam results directory for the full parser.
        dlm_beam_dir (str): beam results directory for the DLM utterance completion from prefix.
        dlm_beam_size (int): beam size for the DLM prefix completion decoding.
        data_sets (List[str]): individual absolute length prefix files.
        data_folder (str): path of the oracle folder that stores the individual absolute prefix length file and the
            indices of the original data that it includes.
        only_dlm (bool, optional): only collect results for the language completion but no parsing results.
            Defaults to False.
        data_subset (str, optional): test data subset. Defaults to 'valid'.
        model (str, optional): [description]. Defaults to 'checkpoint_last'.
        dlm_model (str, optional): [description]. Defaults to 'checkpoint_best'.

    NOTE we assume the `data_sets` contains consecutive prefix lengths, and it has to start from 0 length.
    """

    # ===== collect indexes of incremental prefixes for each utterance
    index_in_stacked_file, prefix_idxs_all = extract_prefix_linenos_in_file(data_sets, data_folder)
    print('index in the stacked prefix file collected!')

    # ===== read in the DLM utterance completion results
    # NOTE the file names are e.g.
    #      for beam 5, "valid_checkpoint_best.hypo" "valid_checkpoint_best.1.hypo" ... "valid_checkpoint_best.4.hypo"
    dlm_completion_results = {}
    dlm_completion_results['src'] = read_string_sentences(
        os.path.join(dlm_beam_dir, f'{data_subset}_{dlm_model}.src')
        )
    dlm_completion_keys = ['hypo', 'step_scores']
    for key in dlm_completion_keys:
        dlm_completion_results.setdefault(key, []).append(
            read_string_sentences(os.path.join(dlm_beam_dir, f'{data_subset}_{dlm_model}.{key}'))
        )    # e.g. f'{valid}_{checkpoint_best}.hypo'

        if dlm_beam_size > 1:
            for bm in range(dlm_beam_size - 1):
                dlm_completion_results.setdefault(key, []).append(
                    read_string_sentences(os.path.join(dlm_beam_dir, f'{data_subset}_{dlm_model}.{bm + 1}.{key}'))
                )
    print(f'language model completion results with beam {dlm_beam_size} loaded!')

    # `dlm_completion_results` is a dictionary containing all (stacked prefixes) beam search results from DLM completion
    # keys and values
    # 'src': List[str], shared across all beams
    # 'hypo': List[List[str]], where each list is corresponding to a beam
    # 'step_scores': List[List[str]], where each list is corresponding to a beam
    #                (NOTE the scores should be coverted float, and should *log(2) to recover log_e scale)

    # ===== read in the parser generation results
    # data sets prefix obtained from DLM utterance completion with beam search -> used by parser
    dlm_beam_data_sets = [f'{data_subset}-prefixallt-beam{dlm_beam_size}-{x + 1}' for x in range(dlm_beam_size)]

    dlm_beam_parsing_results: List[Dict[int, Dict]] = []    # each one in list is corresponding to a DLM completion beam
    if not only_dlm:
        for d in dlm_beam_data_sets:
            ps = json.load(open(os.path.join(beam_dir, f'{d}_{model}.hypos.json')))
            dlm_beam_parsing_results.append(ps)
    print(f'parser decoding results with beam {beam_size} loaded!')

    # ===== sort and collect all results into a single dictionary
    all_prefix_predictions = {}
    for i, prefix_idxs in tqdm(enumerate(index_in_stacked_file), unit=' examples'):
        prefix_predictions = {}
        num_prefixes = len(prefix_idxs)

        for j, idx in enumerate(prefix_idxs):
            # incremental prefixes for a single example
            for dlm_bm in range(dlm_beam_size):
                # different beam results from DLM
                dlm_bm_predictions = {
                    'completion_src_str': dlm_completion_results['src'][idx],
                    'completion_utter_str': dlm_completion_results['hypo'][dlm_bm][idx],
                    'completion_step_scores': (np.array(list(
                        map(float, dlm_completion_results['step_scores'][dlm_bm][idx].split()))
                                                        )
                                               * math.log(2)
                                               ).tolist(),
                    # NOTE the dumped score from DLM is log2 scale -> we recover to loge here to be consistent with
                    #      parsing results
                }
                # parsing results from the utterance completion
                if not only_dlm:
                    dlm_bm_predictions['parsing_results'] = dlm_beam_parsing_results[dlm_bm][str(idx)]

                prefix_predictions.setdefault(j, []).append(
                    dlm_bm_predictions)    # List[Dict], each one is for one beam

        all_prefix_predictions.setdefault(i, {})['predictions'] = prefix_predictions
        # below `j` is the last prefix index, which is the case of full utterance
        all_prefix_predictions[i]['src_str'] = prefix_predictions[j][0]['completion_src_str']    # [0] is for first beam
        assert '<mask>' not in all_prefix_predictions[i]['src_str'], 'seems not full utterance here'
        # target reference: we don't have that actually (we didn't process it to be saved in data)
        # -> it's likely to be <<unk>>
        if not only_dlm:
            all_prefix_predictions[i]['reference'] = prefix_predictions[j][0]['parsing_results'][0]['reference']
        # extract context and utterance
        context = all_prefix_predictions[i]['src_str'].split('__User')[:-1]
        utterance = all_prefix_predictions[i]['src_str'].split('__User')[-1]
        utterance = utterance.split()
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
    #           'completion_src_str': str,      # source input to the language completion model
    #           'completion_utter_str': str,    # completed utterance
    #           'completion_step_scores': List[float],    # completion step-wise log-likelihood
    #           'parsing_results' (optional): List[
    #                               {
    #                               'actions': List[str],
    #                               'step_scores': List[float],
    #                               ...
    #                               }
    #                                  ]    # for different parsing beams
    #           }
    #              ]        # for different utterance completion  beams
    #       1: List[
    #           {
    #           'completion_src_str': str,      # source input to the language completion model
    #           'completion_utter_str': str,    # completed utterance
    #           'completion_step_scores': List[float],    # completion step-wise log-likelihood
    #           'parsing_results' (optional): List[
    #                               {
    #                               'actions': List[str],
    #                               'step_scores': List[float],
    #                               ...
    #                               }
    #                                  ]    # for different parsing beams
    #           }
    #              ]        # for different utterance completion  beams
    #       # for different prefix length
    #       }
    #   'src_str': str,
    #   'context': str,
    #   'utterance': str,
    #   'len_utter': int,
    #   ...
    #   }

    return all_prefix_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge and sort all beam decoding results for language completion -> parsing'
                    'with every prefix length for validation data')
    parser.add_argument('--rootdir', type=str, default=rootdir,
                        help='Root directory of experiment folders')
    parser.add_argument('--subset', type=str, default=subset,
                        help='test data subset')
    parser.add_argument('--max_len', type=int, default=max_len,
                        help='max length in the test data')
    parser.add_argument('--data_folder', type=str, default=data_folder,
                        help='Data folder that stores the prefix source files for DLM')
    parser.add_argument('--dlm_model_folder', type=str, default=dlm_model_folder,
                        help='DLM model experiment folder prefix to be matched')
    parser.add_argument('--dlm_beam_size', type=int, default=dlm_beam_size,
                        help='DLM decoding beam size')
    parser.add_argument('--dlm_model', type=str, default=dlm_model,
                        help='DLM model checkpoint names to collect scores')
    parser.add_argument('--model_folder', type=str, default=model_folder,
                        help='parsing model experiment folder prefix to be matched')
    parser.add_argument('--beam_size', type=int, default=beam_size,
                        help='parsing decoding beam size')
    parser.add_argument('--model', type=str, default=model,
                        help='parsing model checkpoint names to collect scores')
    parser.add_argument('--only_dlm', type=int, default=only_dlm,
                        help='only collecting language completion results')
    # parser.add_argument('--save_path', type=str, default=None,
    #                     help='Path to save the results')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    data_sets = [f'{args.subset}-prefix{x}t' for x in range(args.max_len + 1)]

    # data folder for the prefix -> full utterance model for the language completion model
    data_folder = os.path.join(args.rootdir, args.data_folder)

    # language completion model
    dlm_beam_dir = os.path.join(args.rootdir, args.dlm_model_folder, f'beam{args.dlm_beam_size}')

    # parsing model
    beam_dir = os.path.join(args.rootdir, args.model_folder, f'beam{args.beam_size}')

    all_prefix_predictions = merge_prefix_dlm_to_parser_results(
        beam_dir=beam_dir,
        dlm_beam_dir=dlm_beam_dir,
        dlm_beam_size=args.dlm_beam_size,
        data_sets=data_sets,
        data_folder=data_folder,
        data_subset=args.subset,
        model=args.model,
        dlm_model=args.dlm_model,
        only_dlm=bool(args.only_dlm),
    )

    if args.only_dlm:
        save_path = os.path.join(dlm_beam_dir, f'{args.subset}-prefixallt_{args.dlm_model}.hypos.json')
    else:
        save_path = os.path.join(beam_dir, f'{args.subset}-prefixallt.parsing-from-dlm_{args.dlm_model}-beam{args.dlm_beam_size}.hypos.json')

    print('Saving results ...')
    json.dump(all_prefix_predictions, open(save_path, 'w'))
    if args.only_dlm:
        print(f'All {args.subset} prefix prediction results with DLM beam size {args.dlm_beam_size} saved '
              f'to: {save_path}')
    else:
        print(f'All {args.subset} prefix prediction results with DLM beam size {args.dlm_beam_size} and parser '
              f'beam size {args.beam_size} saved to: {save_path}')
