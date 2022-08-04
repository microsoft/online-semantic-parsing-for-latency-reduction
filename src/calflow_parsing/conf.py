# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from pathlib import Path

# path to `incremental-interpretation` assuming directory structure
# `incremental-interpretation/code/src/calflow_parsing/conf.py`
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_DIR / "DATA"
SAVE_DIR = PROJECT_DIR / "SAVE"
CALFLOW_DIR = DATA_DIR / "smcalflow2.0"
# ASR output for just speech, no accompanying programs
ASR_NO_PROGRAMS_DIR = DATA_DIR / "fdr"
# voice-acted subset of dev
VOICE_ACTED_DIR = DATA_DIR / "dev_300_uniform_audio"
# ASR output for voice-acted subset of dev
VOICE_ACTED_ASR_DIR = VOICE_ACTED_DIR / "asr"

# ASR output for voice-acted dev100 and test200 data
VOICE_ACTED_ASR_DIR = DATA_DIR / 'smcalflow2.0_voice300'
