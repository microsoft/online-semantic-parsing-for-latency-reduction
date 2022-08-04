# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys

from fairseq.data import Dictionary

from fairseq_gap.actions_preprocess.action_data_binarize_scp import (binarize_actstates_tofile,
                                                                     binarize_actstates_tofile_workers,
                                                                     load_actstates_fromfile)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    else:
        num_workers = 1

    split = 'valid'
    # split = 'train'

    en_file = f'/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/act-top_src-ct1-npwa_tgt-cp-str/oracle/{split}.utters'
    actions_file = f'/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/act-top_src-ct1-npwa_tgt-cp-str/oracle/{split}.actions.src_copy'
    actions_dict = Dictionary.load(
        '/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/act-top_src-ct1-npwa_tgt-cp-str/fairseq_bin/dict.actions_nopos.txt'
    )

    out_file_pref = f'/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/tmp/{split}.en-actions.actions'

    os.makedirs(os.path.dirname(out_file_pref), exist_ok=True)

    # binarize_actstates_tofile(en_file, actions_file, out_file_pref, actions_dict=actions_dict)
    res = binarize_actstates_tofile_workers(en_file, actions_file, out_file_pref, actions_dict=actions_dict,
                                            num_workers=num_workers)

    print(
        "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
            'actions',
            actions_file,
            res['nseq'],
            res['ntok'],
            100 * res['nunk'] / (res['ntok'] + 1e-6),    # when it is not recorded: denominator being 0
            actions_dict.unk_word,
        )
    )

    os.system(f'ls -lh {os.path.dirname(out_file_pref)}')

    tgt_actstates = load_actstates_fromfile(out_file_pref, actions_dict)

    breakpoint()
