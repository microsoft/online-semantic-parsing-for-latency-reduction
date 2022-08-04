# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def tokenize_line_tab(line):
    line = line.strip()
    return line.split('\t')
