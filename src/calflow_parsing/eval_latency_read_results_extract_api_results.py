# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Read the latency results and extract improvements by each API call.
"""
import argparse
import json
import pickle

import numpy as np
from tqdm import tqdm

from calflow_parsing.calflow_graph import graph_to_lispress, lispress_to_graph
from calflow_parsing.word_rate import PER_SEC


results_path = (
    '../SAVE'
    '/exp_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
    '/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2'
    '/beam1'
    '/valid-prefixallt'
)
threshold = -0.2    # likely to have not been updated to the correct results (previous run with buggy code)!
lay_file = f'.wordrate-constant.exec1000.threshold{threshold}.lay'
latency_results = results_path + lay_file


results_path = (
        '../SAVE'
        '/exp_prefix-last-8ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
        '/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2'
        '/beam1'
        '/valid-prefixallt'
    )
threshold = -10
threshold = -1.4
threshold = -1
# threshold = -0.2    # likely to have not been updated to the correct results (previous run with buggy code)!
lay_file = f'.wordrate-constant.exec1000.threshold{threshold}.lay'
latency_results = results_path + lay_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latency_results', type=str, default=latency_results)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.latency_results, 'r') as fid:
        policy_records = [json.loads(line.strip()) for line in fid if line.strip()]

    policy_keys = list(filter(lambda x: 'Policy' in x, list(policy_records[0].keys())))
    # include 'WaitUntilEndPolicy()' and 'LogProbThresholdPolicy(threshold=-0.2)'

    baseline_name = ''
    policy_name = ''
    assert len(policy_keys) == 2
    for k in policy_keys:
        if k.startswith('Wait'):
            baseline_name = k
        else:
            policy_name = k

    # record the latency improvements for different API names
    api_lay_imps = {}
    api_lay_baseline = {}

    no_matched_lispress = 0    # not matched lispress
    total_num = 0    # total number of API calls

    for pr in tqdm(policy_records):
        # breakpoint()

        for api_call in pr[baseline_name]['api_calls']:    # pr[baseline_name] with keys "api_calls" "latency"
            # api_call is a dict with keys "lispress", 'start_time', 'finish_time'
            lispress = api_call['lispress']
            finish_time = api_call['finish_time']

            graph = lispress_to_graph(lispress)
            api_name = graph.nodes[graph.root]

            api_lay_baseline.setdefault(api_name, []).append(finish_time / PER_SEC)    # convert to second
            total_num += 1

            # search for same call in the policy
            found_match = False
            for policy_api_call in pr[policy_name]['api_calls']:
                if policy_api_call['lispress'] == lispress:
                    # matched API call
                    # if policy_api_call['finish_time'] < 0:
                    #     # just check if the time is capped at 0 --> Yes it is the case
                    #     breakpoint()
                    imp = finish_time - policy_api_call['finish_time']
                    assert imp >= 0, 'latency improvement should not be negative'
                    api_lay_imps.setdefault(api_name, []).append(imp / PER_SEC)    # convert to second
                    break
            else:
                # print('Did not find matched lispress -- might be a conversion issue')
                # print(lispress)
                no_matched_lispress += 1
                # breakpoint()

                # search again to match graph instead, as lispress may have different format
                for policy_api_call in pr[policy_name]['api_calls']:
                    if graph.is_identical(lispress_to_graph(policy_api_call['lispress'])):
                        # matched API call
                        imp = finish_time - policy_api_call['finish_time']
                        assert imp >= 0, 'latency improvement should not be negative'
                        api_lay_imps.setdefault(api_name, []).append(imp / PER_SEC)
                        break
                else:
                    print('Did not find matched graph even -- might be a different issue')
                    print(lispress)
                    no_matched_lispress += 1

                    breakpoint()

    print(f'Total number of API calls: {total_num}')
    print(f'Number of unmatched lispress (but matched graph): {no_matched_lispress} '
          f'({no_matched_lispress / total_num * 100:.2f}%)')
    print(f'Details for each API call:')
    for k, v in api_lay_imps.items():
        print(f'{k}')
        assert len(v) == len(api_lay_baseline[k])
        print('      improvement: ' + f'{np.mean(v):.2f}' + ' ' * 5 + f'(total num: {len(v)})')
        print('         baseline: ' + f'{np.mean(api_lay_baseline[k]):.2f}')

    # breakpoint()

    # save results for ploting
    save_path = results_path + lay_file + '.apis'
    json.dump({'api_lay_imps': api_lay_imps,
               'api_lay_baseline': api_lay_baseline},
              open(save_path, 'w'))
    print(f'Latency results grouped by API are saved at {save_path}')
