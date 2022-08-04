# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Read the latency results and extract wait-until-end system numbers.
"""
import argparse
import json


threshold = -0.2
latency_results = (
    '../SAVE'
    '/exp_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0'
    '/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2'
    '/beam1'
    '/valid-prefixallt'
    f'.threshold{threshold}.lay'
    )


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
    policy_latencies = {}
    policy_latencies_average = {}
    policy_num_function_calls = {}
    policy_num_function_calls_average = {}
    for k in policy_keys:
        policy_latencies[k] = [pe[k]['latency'] for pe in policy_records]
        policy_latencies_average[k] = sum(policy_latencies[k]) / len(policy_latencies[k])
        policy_num_function_calls[k] = [len(pe[k]['api_calls']) for pe in policy_records]
        policy_num_function_calls_average[k] = sum(policy_num_function_calls[k]) / len(policy_num_function_calls[k])

    # print results
    print('-' * 60)
    print(f'Number of programs / graphs: {len(policy_records)}')
    print('Average Number of function calls', '-' * 10)
    print('\n'.join(f'{k}: {v:.5f}' for k, v in policy_num_function_calls_average.items()))
    print('Average latency of different policies', '-' * 10)
    print('\n'.join(f'{k}: {v:.5f}' for k, v in policy_latencies_average.items()))
    print('-' * 60)
