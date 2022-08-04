# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os


def read_tokenized_sentences(file_path, separator=' '):
    sentences = []
    with open(file_path) as fid:
        for line in fid:
            sentences.append(line.rstrip().split(separator))
    return sentences


def read_string_sentences(file_path):
    with open(file_path, 'r') as fid:
        sentences = [line.strip() for line in fid if line.strip()]
    return sentences


def mkdir_for_file(file_path):
    dir = os.path.abspath(os.path.dirname(file_path))
    os.makedirs(dir, exist_ok=True)
    return


def write_tokenized_sentences(file_path, content, separator=' '):
    mkdir_for_file(file_path)
    with open(file_path, 'w') as fid:
        for line in content:
            line = [str(x) for x in line]
            fid.write(f'{separator.join(line)}\n')


def write_string_sentences(file_path, content):
    mkdir_for_file(file_path)
    with open(file_path, 'w') as fid:
        for line in content:
            assert isinstance(line, str)
            fid.write(line + '\n')
