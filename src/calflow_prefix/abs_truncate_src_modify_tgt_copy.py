# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Truncate a source utterance to a certain length (absolute), and modify the target with _COPY_(src_pointer) actions
to be some special placeholder copy actions without the src_pointer if it hasn't appeared.
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
    parser = argparse.ArgumentParser(description='truncate the src to certain prefix length')
    parser.add_argument('--src_utterances', type=str,
                        help='file path of the tokenized source utterances (with contexts)')
    parser.add_argument('--tgt_actions', type=str,
                        help='file path of the target actions (with _COPY_(src_pointer) actions)')
    parser.add_argument('--out_src_prefix', type=str,
                        help='file path of the output src prefix')
    parser.add_argument('--out_actions', type=str,
                        help='file path of the output tgt actions with unseen copy modified')
    parser.add_argument('--out_indices', type=str,
                        help='file path of the corresponding indices of the output data in the original data file')
    parser.add_argument('--len_abs', type=int, default=5,
                        help='abslute length to truncate to remain in the generated prefix')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    src_file = args.src_utterances
    tgt_file = args.tgt_actions
    out_file = args.out_src_prefix
    out_tgt_file = args.out_actions
    out_idx_file = args.out_indices
    len_abs = args.len_abs

    mkdir_for_file(out_file)
    mkdir_for_file(out_tgt_file)

    with open(src_file, 'r') as f, open(tgt_file, 'r') as ft, \
          open(out_file, 'w') as g, open(out_tgt_file, 'w') as g2, open(out_idx_file, 'w') as g3:
        count = 0
        for line_no, (line, line_tgt) in tqdm(enumerate(zip(f, ft))):
            if line.strip():
                assert line_tgt.strip()

                context = line.strip().split('__User')[:-1]
                utterance = line.strip().split('__User')[-1]
                total_len = len(utterance.split()[:-1])
                if len_abs > total_len:
                    continue

                prefix = get_prefix(utterance.split()[:-1], truncate_len=len_abs)

                if len_abs == total_len:
                    prefix = ' '.join(['__User'] + prefix + ['__StartOfProgram'])
                else:
                    prefix = ' '.join(['__User'] + prefix + ['<mask>', '__StartOfProgram'])
                truncated_utterance = '__User'.join(context) + prefix
                len_src_prefix = len(truncated_utterance.split())

                g.write(truncated_utterance + '\n')

                actions = line_tgt.strip().split()
                actions_updated = []
                for act in actions:
                    if act.startswith(SpecialSymbols.COPY):
                        _, src_pos = peel_copy_src_pointer(act)
                        if src_pos >= len_src_prefix - 1:    # last token is "__StartOfProgram" -> shouldn't be copied
                            # the copy is not reachable yet -> modify the copy action

                            # # directly use a new action
                            # act = SpecialSymbols.COPY_ANONYM

                            # copy the future mask
                            src_pos = len_src_prefix - 2
                            assert truncated_utterance.split()[src_pos] == '<mask>'
                            act = join_copy_src_pointer(SpecialSymbols.COPY, src_pos)

                    actions_updated.append(act)

                actions_updated = ' '.join(actions_updated)
                g2.write(actions_updated + '\n')

                g3.write(str(line_no) + '\n')
                count += 1

    print(f'Number of lines with length {len_abs} written: {count} (total {line_no + 1})')
