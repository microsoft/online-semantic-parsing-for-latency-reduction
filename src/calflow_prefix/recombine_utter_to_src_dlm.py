# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Recombine the predicted full utterance (from the dlm) into the context with prefix to formulate the complete src
for the parser to use.
"""
import argparse

from tqdm import tqdm

from calflow_parsing.io import mkdir_for_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='recombine the predicted full utterance with the context (with prefix) to formulate the complete src to be used by the parser')
    parser.add_argument('--src_utterances', type=str,
                        help='file path of the tokenized source utterances (with contexts)')
    parser.add_argument('--src_completions', type=str,
                        help='file path of the tokenized source utterances (with no contexts)')
    parser.add_argument('--out_src', type=str,
                        help='file path of the output src (context + full utterance)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    with open(args.src_utterances, 'r') as f1, open(args.src_completions, 'r') as f2, \
            open(args.out_src, 'w') as g:
        for src_partial, src_utter in tqdm(zip(f1, f2)):
            if src_partial := src_partial.strip():
                src_utter = src_utter.strip()

                context = src_partial.strip().split('__User')[:-1]
                partial_utterance = src_partial.strip().split('__User')[-1]

                src_utter = src_utter.split()
                assert '__StartOfProgram' not in src_utter and '__User' not in src_utter
                src_utter = ' '.join(['__User'] + src_utter + ['__StartOfProgram'])
                src = '__User'.join(context) + src_utter

                # write completed src
                g.write(src)
                g.write('\n')

    print(f'Complete src sequences written at {args.out_src}')
