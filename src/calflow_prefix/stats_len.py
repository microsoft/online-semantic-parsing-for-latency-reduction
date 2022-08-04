# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Store the lengths information of the dataset into a file."""
import argparse
import json

from tqdm import tqdm

from calflow_parsing.io import mkdir_for_file


def parse_args():
    parser = argparse.ArgumentParser(description='stats of lengths of source utterance (with context) data')
    parser.add_argument('--src_utterances', type=str,
                        help='file path of the tokenized source utterances (with contexts)')
    parser.add_argument('--out_len_stats', type=str,
                        help='file path of the length statistics')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.src_utterances, 'r') as fid:
        lines = [line.strip() for line in fid if line.strip()]

    contexts = []
    utterances = []
    lengths = []
    src_lengths = []
    len11ps_all = []

    # data_sets = [
    #     'valid-prefix100p', 'valid-prefix90p', 'valid-prefix80p', 'valid-prefix70p', 'valid-prefix60p',
    #     'valid-prefix50p', 'valid-prefix40p', 'valid-prefix30p', 'valid-prefix20p', 'valid-prefix10p',
    #     'valid-prefix0p'
    # ]

    data_stats = {}

    for idx, line in tqdm(enumerate(lines)):
        context = line.strip().split('__User')[:-1]
        utterance = line.strip().split('__User')[-1]
        utterance = utterance.split()[:-1]
        total_len = len(utterance)

        contexts.append('__User'.join(context))
        utterances.append(' '.join(utterance))
        lengths.append(total_len)
        src_lengths.append(len(line.strip().split()))

        len11ps = []

        for perc in range(0, 110, 10):
            truncate_len = round(total_len * perc / 100)
            len11ps.append(truncate_len)

        len11ps_all.append(len11ps)

    data_stats['contexts'] = contexts
    data_stats['utterances'] = utterances
    data_stats['lens_utter'] = lengths
    data_stats['lens_src'] = src_lengths
    data_stats['lens_11ps'] = len11ps_all

    mkdir_for_file(args.out_len_stats)

    json.dump(data_stats, open(args.out_len_stats, 'w'))
    print(f'Data lengths statistics saved at {args.out_len_stats}')
    print(f'----- #examples: {len(lengths)} | min len: {min(lengths)} | max len: {max(lengths)} | avg len: {sum(lengths) / len(lengths)}')
