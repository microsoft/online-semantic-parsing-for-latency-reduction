#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e

[[ -f ../DATA/smcalflow2.0_10k/valid.dataflow_dialogues.jsonl ]] && echo "already exists - do not overwrite as we may have run experiments on it" && exit 1

python scripts/random_sample_dialogues.py ../DATA/smcalflow2.0/train.dataflow_dialogues.jsonl 10_000
cp ../DATA/smcalflow2.0/valid.dataflow_dialogues.jsonl ../DATA/smcalflow2.0_10k/valid.dataflow_dialogues.jsonl
