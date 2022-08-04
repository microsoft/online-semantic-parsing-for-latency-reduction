# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np


if __name__ == '__main__':
    src_file = (
        '../DATA'
        '/processed/smcalflow2.0'
        '/act-top_src-ct1-npwa_tgt-cp-str'
        '/oracle/'
        '/valid.utters'
    )
    src_file = (
        '../DATA'
        '/processed/smcalflow2.0'
        '/old_act-top_src-ct1-npwa_tgt-cp-str_prefix-all'
        '/oracle/'
        '/valid-prefix0p.utters'
    )
    tgt_file = (
        '../DATA'
        '/processed/smcalflow2.0'
        '/act-top_src-ct1-npwa_tgt-cp-str'
        '/oracle/'
        '/valid.actions'
    )
    src_lens = []
    tgt_lens = []
    ratios = []
    with open(src_file, 'r') as f, open(tgt_file, 'r') as g:
        for l1, l2 in zip(f, g):
            if l1.strip():
                assert l2.strip()
                len1 = len(l1.strip().split())
                len2 = len(l2.strip().split())
                ratio = len2 / len1
                # ratio = (len2 - 200) / len1

                src_lens.append(len1)
                tgt_lens.append(len2)
                ratios.append(ratio)

    print('src lens (0..10..100 percentiles):')
    print(np.percentile(src_lens, range(0, 110, 10)).tolist())

    print('tgt lens (0..10..100 percentiles):')
    print(np.percentile(tgt_lens, range(0, 110, 10)).tolist())

    print('tgt/src len ratios (0..10..100 percentiles):')
    print(np.percentile(ratios, range(0, 110, 10)).tolist())