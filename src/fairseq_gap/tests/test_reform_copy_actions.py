# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from re import T
import sys
import os

from fairseq_gap.actions_interface.source_copy import CalFlowActionsSrcCopyInterface


if __name__ == '__main__':
    if len(sys.argv) > 1:
        split = sys.argv[1]
    else:
        split = 'valid'

    en_file = f'/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/act-top_src-ct0-npna/oracle/{split}.utters'
    actions_file = f'/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/act-top_src-ct0-npna/oracle/{split}.actions'

    en_file = f'/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/act-top_src-ct1-npwa/oracle/{split}.utters'
    actions_file = f'/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/act-top_src-ct1-npwa/oracle/{split}.actions'

    out_file_pref = f'/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/tmp/{split}.actions'

    os.makedirs(os.path.dirname(out_file_pref), exist_ok=True)

    only_string = True               # on valid data, avg copy is 0.95 when True or 1.11 when False
    only_from_current_utter = True    # on valid data, less impact (0.95 vs 0.95, 1.11 vs 1.12)

    num_avg_copies, num_total_copies, num_lines = \
        CalFlowActionsSrcCopyInterface.reform_actions_with_copy_file(
            en_file,
            actions_file,
            out_file_pref = out_file_pref,
            only_string=only_string,
            only_from_current_utter=only_from_current_utter
            )

    print(f'Number of average copies per action sequence: {num_avg_copies:.2f} ({num_total_copies} / {num_lines})')
