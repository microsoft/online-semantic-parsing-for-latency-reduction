# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Check for repetitive tokens at consecutive steps."""
from collections import Counter
from itertools import chain
import sys


def extract_repetitive_tokens(line):
    if isinstance(line, str):
        line = line.split()
    elif isinstance(line, list):
        assert isinstance(line[0], str)
    else:
        raise ValueError

    rep_idxs = []
    rep_toks = []
    for i, (a, b) in enumerate(zip(line, line[1:])):
        if a == b:
            rep_idxs.append(i)
            rep_toks.append(a)

    return rep_idxs, rep_toks


if __name__ == '__main__':
    file = sys.argv[1]
    with open(file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    rep_lines = []
    rep_tokens = []
    num_rep_tokens = 0
    for i, line in enumerate(lines, start=1):
        rep_idxs, rep_toks = extract_repetitive_tokens(line)
        if rep_idxs:
            rep_lines.append(i)
            rep_tokens.append(rep_toks)
            num_rep_tokens += len(rep_toks)

    rep_tokens_counter = Counter(chain.from_iterable(rep_tokens))
    print(f'Number of lines with repetitive tokens: {len(rep_lines)} / {len(lines)} ({len(rep_lines) / len(lines):.5f})')
    print('Counter of repetitive tokens:')
    print(rep_tokens_counter)
