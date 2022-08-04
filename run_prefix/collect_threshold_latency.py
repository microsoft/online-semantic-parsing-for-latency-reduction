# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Collect latency results obtained by scanning through a policy threshold on graph predictions,
corresponding to a latency_improvement vs. cost curve.
"""

import argparse
from datetime import datetime
import os
import pickle
import re

import numpy as np


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
# model_folder = ('exp_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
#                 '/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2')

latency_results_dir = os.path.join(rootdir, model_folder, 'beam1')

word_rate_model = 'constant'
execution_time = 0


# [template] latency improvement summary file name regex
file_lay_re = re.compile(r'valid-prefixallt.threshold(.+?).lay.summary')

# latency improvement summary file content regex
# lay_imp_re = re.compile(r'avg_latency_improvement,(.+)')
# num_call_re = re.compile(r'avg_num_api_calls,(.+)')
lay_imp_re = re.compile(r'avg_latency_improvement: (.+)')
num_call_re = re.compile(r'avg_num_api_calls: (.+)')
lay_wait_until_end_re = re.compile(r'avg_WaitUntilEnd_latency: (.+)')
num_call_wait_until_end_re = re.compile(r'avg_num_WaitUntilEnd_api_calls: (.+)')
lay_imp_oracle_re = re.compile(r'avg_latency_improvement_oracle: (.+)')
lay_imp_nocap_re = re.compile(r'avg_latency_improvement_nocap: (.+)')
lay_nocap_re = re.compile(r'avg_latency_nocap: (.+)')


def collect_lay_summary(search_dir, file_lay_re):
    # search all files with latency result summaries to extract results automatically
    thresholds = []
    lay_imps = []
    num_calls = []
    lay_imps_nocap = []
    lay_nocap = []
    lay_imp_oracle = 0.0    # in case it does not exist in file (we didn't record it in previous runs)
    for dfile in os.listdir(search_dir):
        if not file_lay_re.match(dfile):
            continue

        # add a file creation time filter to get the latest correct runs
        if not datetime.fromtimestamp(os.path.getmtime(os.path.join(search_dir, dfile))) > datetime(2022, 3, 10):
            continue

        threshold, = file_lay_re.match(dfile).groups()
        threshold = float(threshold)
        thresholds.append(threshold)

        with open(os.path.join(search_dir, dfile), 'r') as fid:
            for line in fid:
                if lay_imp_re.match(line):
                    lay_imp, = lay_imp_re.match(line).groups()
                    lay_imps.append(float(lay_imp))
                if num_call_re.match(line):
                    num_call, = num_call_re.match(line).groups()
                    num_calls.append(float(num_call))

                if lay_wait_until_end_re.match(line):
                    lay_wait_until_end, = lay_wait_until_end_re.match(line).groups()
                    lay_wait_until_end = float(lay_wait_until_end)
                if num_call_wait_until_end_re.match(line):
                    num_calls_wait_until_end, = num_call_wait_until_end_re.match(line).groups()
                    num_calls_wait_until_end = float(num_calls_wait_until_end)

                if lay_imp_oracle_re.match(line):
                    lay_imp_oracle, = lay_imp_oracle_re.match(line).groups()
                    lay_imp_oracle = float(lay_imp_oracle)

                if lay_imp_nocap_re.match(line):
                    lay_imp_nocap, = lay_imp_nocap_re.match(line).groups()
                    lay_imps_nocap.append(float(lay_imp_nocap))
                if lay_nocap_re.match(line):
                    lnocap, = lay_nocap_re.match(line).groups()
                    lay_nocap.append(float(lnocap))

        assert len(thresholds) == len(lay_imps) == len(num_calls)
        assert len(thresholds) == len(lay_imps_nocap) == len(lay_nocap)

    # sort the results
    sort_indices = np.argsort(thresholds)[::-1]
    thresholds = np.array(thresholds)[sort_indices]
    lay_imps = np.array(lay_imps)[sort_indices]
    num_calls = np.array(num_calls)[sort_indices]
    lay_imps_nocap = np.array(lay_imps_nocap)[sort_indices]
    lay_nocap = np.array(lay_nocap)[sort_indices]

    return lay_imps, num_calls, thresholds, lay_wait_until_end, num_calls_wait_until_end, lay_imps_nocap, lay_nocap, \
        lay_imp_oracle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latency-results-dir',
                        type=str,
                        default=latency_results_dir,
                        help='directory of latency results (usually MODEL_FOLDER / beam1')
    parser.add_argument('--word-rate-model',
                        type=str,
                        choices=['constant', 'char-linear', 'real-voice'],    # 'constant' is the intrinsic setting
                        default='char-linear',
                        help='word speaking rate scheme',
                        )
    parser.add_argument('--execution-time',
                        type=float,
                        default=execution_time,
                        help='execution time of a function node (in millisecond)',
                        )
    parser.add_argument('--output-tag',
                        type=str,
                        default='valid-prefixallt',
                        help='output file prefix tag')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # NOTE float type in args would include .0 in the string; we remove it for all integers in file names
    if int(args.execution_time) == args.execution_time:
        args.execution_time = int(args.execution_time)

    # latency improvement summary file name regex
    file_lay_re = re.compile(
        rf'{args.output_tag}.wordrate-{args.word_rate_model}.exec{args.execution_time}.threshold(.+?).lay.summary'
        )

    lay_imps, num_calls, thresholds, lay_wait_until_end, num_calls_wait_until_end, \
        lay_imps_nocap, lay_nocap, lay_imp_oracle = collect_lay_summary(args.latency_results_dir, file_lay_re)

    save_results = {
        'word_rate_model': args.word_rate_model,
        'execution_time': args.execution_time,
        'lay_imps': lay_imps,
        'num_calls': num_calls,
        'thresholds': thresholds,
        'lay_wait_until_end': lay_wait_until_end,
        'num_calls_wait_until_end': num_calls_wait_until_end,
        'lay_imp_oracle': lay_imp_oracle,
        'lay_imps_nocap': lay_imps_nocap,
        'lay_nocap': lay_nocap,
    }
    save_path = os.path.join(
        args.latency_results_dir,
        f'{args.output_tag}.wordrate-{args.word_rate_model}.exec{args.execution_time}.lay-collection.pkl'
        )

    pickle.dump(save_results, open(save_path, 'wb'))

    print()
    print(f'Word rate model: {args.word_rate_model}')
    print(f'Execution time: {args.execution_time}')
    print('thresholds:', thresholds)
    print('latency improvements:', lay_imps)
    print('number of calls:', num_calls)
    print('latency wait-until-end:', lay_wait_until_end)
    print('number of calls gold:', num_calls_wait_until_end)
    print('oracle latency improvements:', lay_imp_oracle)
    print()
    print('Latency results under different thresholds are saved at ---')
    print(f'{save_path}')
    print()
