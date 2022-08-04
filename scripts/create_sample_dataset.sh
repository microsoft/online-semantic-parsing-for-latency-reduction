#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e

. set_default_dirs.sh

[[ -f $DATADIR/smcalflow2.0_sample/valid.dataflow_dialogues.jsonl ]] && echo "already exists - do not overwrite as we may have run experiments on it" && exit 1

python scripts/random_sample_dialogues.py $DATADIR/smcalflow2.0/train.dataflow_dialogues.jsonl 100 $DATADIR/smcalflow2.0_sample
python scripts/random_sample_dialogues.py $DATADIR/smcalflow2.0/valid.dataflow_dialogues.jsonl 50 $DATADIR/smcalflow2.0_sample
