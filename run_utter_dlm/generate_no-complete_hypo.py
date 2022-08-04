# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Compute the hypos by treating the target as the source prefix, without doing any
prefix completion to the full utterance.
"""
from tqdm import tqdm


if __name__ == '__main__':
    src_prefix_utter_file = (
        # '/mnt/container_amulet/DATA2/processed/smcalflow2.0/'
        '/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/treedst/'
        'src-ct1-npwa_utter-dlm_abs-prefix-all/oracle/valid.utters')
    tgt_prefix_hypo_file = (
        # '/mnt/container_amulet/DATA2/processed/smcalflow2.0/'
        '/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/treedst/'
        'src-ct1-npwa_utter-dlm_abs-prefix-all/oracle/valid.utters.prefix')
    with open(src_prefix_utter_file, 'r') as f, open(tgt_prefix_hypo_file, 'w') as g:
        for line in tqdm(f):
            if line := line.strip():
                context = line.strip().split('__User')[:-1]
                utterance = line.strip().split('__User')[-1]
                if utterance.split()[-1] == '<mask>':
                    prefix = ' '.join(utterance.split()[:-1])
                else:
                    prefix = utterance

                g.write(prefix)
                g.write('\n')

    print(f'Prefix as target hypo written to {tgt_prefix_hypo_file}')
