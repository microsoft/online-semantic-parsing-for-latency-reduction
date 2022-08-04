# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Randomly sample training dialogues to make the dataset smaller."""
import sys
import random
from pathlib import Path


if __name__ == '__main__':
    file = sys.argv[1]
    size = sys.argv[2]

    if len(sys.argv) >= 4:
        new_dir = sys.argv[3]
    else:
        new_dir = None

    file_path = Path(file)
    size = int(size)
    random.seed(0)

    if new_dir is None:
        if size >= 1000:
            new_dir = str(file_path.parent) + f'_{round(size / 1000)}k'
        else:
            new_dir = str(file_path.parent) + f'_{size}'
    # new_file = file_path.stem + f'_{round(size / 1000)}k' + file_path.suffix
    new_file = file_path.stem + '' + file_path.suffix
    new_file_path = Path(new_dir, new_file)

    new_file_path.parent.mkdir(parents=True, exist_ok=True)

    if size >= 1000:
        sampled_ids_file = file_path.stem + f'_{round(size / 1000)}k' + '.sampled_ids'
    else:
        sampled_ids_file = file_path.stem + f'_{size}' + '.sampled_ids'
    sampled_ids_file_path = Path(new_dir, sampled_ids_file)

    with file_path.open('r') as f:
        lines = [line.strip() for line in f if line.strip()]

    total_size = len(lines)
    sampled_ids = random.sample(range(total_size), size)

    sampled_lines = [lines[x] for x in sampled_ids]

    sampled_ids_file_path.write_text('\n'.join(map(str, sampled_ids)) + '\n')
    new_file_path.write_text('\n'.join(sampled_lines) + '\n')

    print(f'{size} randomly sampled lines from [{file}] is written to:')
    print(f'[{new_file_path}]')
    print(f'(sampled ids (0 indexed) are recorded at [{sampled_ids_file_path}])')
