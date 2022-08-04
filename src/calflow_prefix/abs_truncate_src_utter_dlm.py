# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Truncate a source utterance to a certain length (absolute), and pair it with the original full utterance, for
the purpose of a denoising language model.
"""
import argparse

from tqdm import tqdm

from calflow_parsing.io import mkdir_for_file
from fairseq_gap.constants import SpecialSymbols
from fairseq_gap.actions_interface.source_copy import peel_copy_src_pointer, join_copy_src_pointer


def get_prefix(utterance, truncate_len=None):
    """Get the prefix of an utterance."""
    assert type(utterance) == list
    assert utterance[-1] != '__StartOfProgram'
    total_len = len(utterance)
    if truncate_len is None:
        truncate_len = total_len
    prefix = utterance[:truncate_len]
    return prefix


def parse_args():
    parser = argparse.ArgumentParser(
        description='truncate the src to certain prefix length and pair with original utterance')
    parser.add_argument('--src_utterances', type=str,
                        help='file path of the tokenized source utterances (with contexts)')
    parser.add_argument('--out_src_prefix', type=str,
                        help='file path of the output src prefix')
    parser.add_argument('--out_tgt_utterance', type=str,
                        help='file path of the output tgt full utterances')
    parser.add_argument('--out_indices', type=str,
                        help='file path of the corresponding indices of the output data in the original data file')
    parser.add_argument('--len_abs', type=int, default=5,
                        help='abslute length to truncate to remain in the generated prefix')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    src_file = args.src_utterances
    out_file = args.out_src_prefix
    out_tgt_file = args.out_tgt_utterance
    out_idx_file = args.out_indices
    len_abs = args.len_abs

    mkdir_for_file(out_file)
    mkdir_for_file(out_tgt_file)

    with open(src_file, 'r') as f, \
          open(out_file, 'w') as g, open(out_tgt_file, 'w') as g2, open(out_idx_file, 'w') as g3:
        count = 0
        for line_no, line in tqdm(enumerate(f)):
            if line.strip():

                context = line.strip().split('__User')[:-1]
                utterance = line.strip().split('__User')[-1]
                total_len = len(utterance.split()[:-1])
                if len_abs > total_len:
                    continue

                prefix = get_prefix(utterance.split()[:-1], truncate_len=len_abs)

                if len_abs == total_len:
                    # prefix = ' '.join(['__User'] + prefix + ['__StartOfProgram'])
                    prefix = ' '.join(['__User'] + prefix)
                else:
                    # prefix = ' '.join(['__User'] + prefix + ['<mask>', '__StartOfProgram'])
                    prefix = ' '.join(['__User'] + prefix + ['<mask>'])
                truncated_utterance = '__User'.join(context) + prefix
                len_src_prefix = len(truncated_utterance.split())

                g.write(truncated_utterance + '\n')

                # g2.write(utterance + '\n')    # this includes '__StartOfProgram'
                g2.write(' '.join(utterance.split()[:-1]) + '\n')

                g3.write(str(line_no) + '\n')
                count += 1

    print(f'Number of lines with length {len_abs} written: {count} (total {line_no + 1})')
