# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Compute the BLEU score by treating the target as the source prefix, without doing any
prefix completion to the full utterance.
"""
from fairseq import scoring
from fairseq.scoring import bleu
from fairseq.data import Dictionary, encoders
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from omegaconf import OmegaConf
from tqdm import tqdm


if __name__ == '__main__':
    # build dictionary
    dict_path = (
        # '/mnt/container_amulet/DATA2/processed/smcalflow2.0/'
        '/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/treedst/'
        'src-ct1-npwa_utter-dlm_abs-prefix-all/fairseq_bin/dict.txt')
    tgt_dict = Dictionary.load(dict_path)

    # build scorer
    cfg_scoring = {'_name': 'bleu', 'pad': 1, 'eos': 2, 'unk': 3}
    cfg_scoring = OmegaConf.create(cfg_scoring)    # type: omegaconf.DictConf

    # NOTE this returns None -> does not work! (related to the cfg config)
    # scorer = scoring.build_scorer(cfg_scoring, tgt_dict)
    scorer = bleu.Scorer(
        bleu.BleuConfig(pad=tgt_dict.pad(), eos=tgt_dict.eos(), unk=tgt_dict.unk())
    )

    # build gpt2 bpe
    cfg_bpe = {'_name': 'gpt2',
               'gpt2_encoder_json': 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json',
               'gpt2_vocab_bpe': 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'}
    cfg_bpe = OmegaConf.create(cfg_bpe)    # type: omegaconf.DictConf

    # NOTE this returns None -> does not work! (related to the cfg config)
    # bpe = encoders.build_bpe(cfg_bpe)
    bpe = GPT2BPE(cfg_bpe)

    # read in data and evaluate
    target_path = (
        # '/mnt/container_amulet/DATA2/processed/smcalflow2.0/'
        '/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/treedst/'
        'src-ct1-npwa_utter-dlm_abs-prefix-all/oracle/valid.fullutters')
    hypo_path = (
        # '/mnt/container_amulet/SAVE2/'
        # 'exp_utter_abs-prefix-all_dlm_bart-large/models_ep1_seed42_fp16-lr3e-05-schedulerinverse_sqrt-mt2048x4-wm500-dp0.1-cn0.1/'
        '/n/tata_ddos_ceph/jzhou/incremental-interpretation/SAVE/'
        'exp_treedst_utter_abs-prefix-all_dlm_bart-large/models_ep12_seed42_fp16-lr5e-05-schedulerpolynomial_decay-mt2048x4-wm500-dp0.1-cn0.1/'
        'beam1/valid_checkpoint_last.hypo'
        )
    hypo_path = (
        # '/mnt/container_amulet/DATA2/processed/smcalflow2.0/'
        '/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/treedst/'
        'src-ct1-npwa_utter-dlm_abs-prefix-all/oracle/valid.utters.prefix'
        )

    with open(target_path, 'r') as f, open(hypo_path, 'r') as g:
        for detok_target_str, hypo_str in tqdm(zip(f, g)):
            if detok_target_str := detok_target_str.strip():
                # should be bpe str
                target_str = bpe.encode(detok_target_str)

                # detok_hypo_str = target_str
                detok_hypo_str = hypo_str.strip()
                hypo_str = bpe.encode(detok_hypo_str)

                # breakpoint()

                target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                # hypo_tokens = target_tokens
                hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
                # torch.IntTensor expected by the BLEU scorer

                if hasattr(scorer, "add_string"):    # not True for the current BLEU parser
                    scorer.add_string(target_str, hypo_str)
                else:
                    scorer.add(target_tokens, hypo_tokens)

    print(scorer.result_string())
