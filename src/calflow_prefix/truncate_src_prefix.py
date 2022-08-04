# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Truncate a tokenized src utterance, with turn contexts, to a certain length based on a percentage."""
import argparse

from tqdm import tqdm


def get_prefix(utterance, perc=50):
    """Get the prefix of an utterance."""
    assert type(utterance) == list
    assert utterance[-1] != '__StartOfProgram'
    total_len = len(utterance)
    truncate_len = round(total_len * perc / 100)
    prefix = utterance[:truncate_len]
    return prefix


def parse_args():
    parser = argparse.ArgumentParser(description='truncate the src to certain prefix length')
    parser.add_argument('--src_utterances', type=str,
                        help='file path of the tokenized source utterances (with contexts)')
    parser.add_argument('--out_src_prefix', type=str,
                        help='file path of the output src prefix')
    parser.add_argument('--len_percentage', type=int, default=50,
                        help='percentage of length to truncate to remain in the generated prefix')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    src_file = args.src_utterances
    out_file = args.out_src_prefix
    perc = args.len_percentage

    with open(src_file, 'r') as f, open(out_file, 'w') as g:
        for line in tqdm(f):
            if line.strip():
                context = line.strip().split('__User')[:-1]
                utterance = line.strip().split('__User')[-1]
                prefix = get_prefix(utterance.split()[:-1], perc=perc)
                prefix = ' '.join(['__User'] + prefix + ['__StartOfProgram'])
                truncated_utterance = '__User'.join(context) + prefix
                g.write(truncated_utterance + '\n')
