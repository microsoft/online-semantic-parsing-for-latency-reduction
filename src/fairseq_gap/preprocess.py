#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import logging
import os
import shutil
import sys
from collections import Counter
from itertools import zip_longest
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from fairseq import options, tasks, utils
from fairseq.binarizer import Binarizer
# from fairseq.data import indexed_dataset
from fairseq.tokenizer import tokenize_line

from fairseq_gap import options
from fairseq_gap.utils_import import import_user_module
from fairseq_gap.data import indexed_dataset
from fairseq_gap.pretrained_features.binarize_pretrained import make_pretrained_features



logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.preprocess")


def main(args):
    import_user_module(args)

    os.makedirs(args.destdir, exist_ok=True)
    os.makedirs(args.embdir, exist_ok=True)

    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(args.destdir, "preprocess.log"),
        )
    )
    logger.info(args)

    # to control what preprocessing needs to be run (as they take both time and storage so we avoid running repeatedly)
    run_basic = True
    # this includes:
    # src: build src dictionary, copy the raw data to dir; build src binary data (need to refactor later if this is not needed)
    # tgt: split target pointer values into a separate file; build tgt dictionary, binarize the actions and pointer values
    run_act_states = True
    # this includes:
    # run the state machine in canonical mode to get states information to facilitate modeling;
    run_pretrained_feat = True
    # this includes:
    # for src sentences, use pre-trained (e.g. RoBERTa) model to extract contextual embeddings for each word;
    # this needs GPU and only needs to run once for the English sentences, which does not change for different oracles;
    # thus the embeddings are stored separately from the oracles.

    if os.path.exists(os.path.join(args.destdir, '.done')):
        logger.info(f'binarized source, actions and states in {args.destdir} already exists; not rerunning.')
        run_basic = False
        run_act_states = False
    if os.path.exists(os.path.join(args.embdir, '.done')):
        logger.info(f'pre-trained embeddings in {args.embdir} already exists; not rerunning.')
        run_pretrained_feat = False

    task = tasks.get_task(args.task)

    # preprocess target actions files, to split '.actions' to '.actions_nopos' and '.actions_pos'
    # when building dictionary on the target actions sequences
    # split the action file into two files, one without arc pointer and one with only arc pointer values
    # and the dictionary is only built on the no pointer actions
    # NOTE dictionary is built here, but the data is re-processed inside task to be used by the model
    if run_basic:
        assert args.target_lang == 'actions', 'target extension must be "actions"'
        actions_files = [f'{pref}.{args.target_lang}'
                         for pref in (args.trainpref, args.validpref, args.testpref)
                         if pref is not None]
        source_files = [f'{pref}.{args.source_lang}'
                        for pref in (args.trainpref, args.validpref, args.testpref)
                        if pref is not None]
        task.preprocess_action_files(actions_files, source_filenames=source_files)
        args.target_lang_nopos = 'actions_nopos'    # only build dictionary without pointer values
        args.target_lang_pos = 'actions_pos'

    # set tokenizer
    tokenize = task.tokenize if hasattr(task, 'tokenize') else tokenize_line

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    target = not args.only_source

    if run_basic:
        # if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        #     raise FileExistsError(dict_path(args.source_lang))
        # if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        #     raise FileExistsError(dict_path(args.target_lang))

        if args.joined_dictionary:
            assert (
                not args.srcdict or not args.tgtdict
            ), "cannot use both --srcdict and --tgtdict with --joined-dictionary"

            if args.srcdict:
                src_dict = task.load_dictionary(args.srcdict)
            elif args.tgtdict:
                src_dict = task.load_dictionary(args.tgtdict)
            else:
                assert (
                    args.trainpref
                ), "--trainpref must be set if --srcdict is not specified"
                src_dict = build_dictionary(
                    {train_path(lang) for lang in [args.source_lang, args.target_lang_nopos]},
                    src=True,
                )
            tgt_dict = src_dict
        else:
            if args.srcdict:
                src_dict = task.load_dictionary(args.srcdict)
            else:
                assert (
                    args.trainpref
                ), "--trainpref must be set if --srcdict is not specified"
                src_dict = build_dictionary([train_path(args.source_lang)], src=True)

            if target:
                if args.tgtdict:
                    tgt_dict = task.load_dictionary(args.tgtdict)
                else:
                    assert (
                        args.trainpref
                    ), "--trainpref must be set if --tgtdict is not specified"
                    tgt_dict = build_dictionary([train_path(args.target_lang_nopos)], tgt=True)
            else:
                tgt_dict = None

        src_dict.save(dict_path(args.source_lang))
        if target and tgt_dict is not None:
            tgt_dict.save(dict_path(args.target_lang_nopos))

    logger.info("Dictionaries for src and actions (no pointer) built and saved in {}".format(args.destdir))

    if args.dict_only:
        return

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        logger.info("[{}] Dictionary: {} types".format(lang, len(vocab)))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                        False,    # note here we shut off append eos
                        tokenize,
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, lang, "bin"),
            impl=args.dataset_impl,
            vocab_size=len(vocab),
            dtype=np.int64,
        )
        merge_result(
            Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t), offset=0, end=offsets[1],
                append_eos=False,
                tokenize=tokenize,
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        logger.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_binary_alignment_dataset(input_prefix, output_prefix, num_workers):
        nseq = [0]

        def merge_result(worker_result):
            nseq[0] += worker_result["nseq"]

        input_file = input_prefix
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize_alignments,
                    (
                        args,
                        input_file,
                        utils.parse_alignment,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, None, "bin"), impl=args.dataset_impl
        )

        merge_result(
            Binarizer.binarize_alignments(
                input_file,
                utils.parse_alignment,
                lambda t: ds.add_item(t),
                offset=0,
                end=offsets[1],
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, None)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))

        logger.info("[alignments] {}: parsed {} alignments".format(input_file, nseq[0]))

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1, dataset_impl=args.dataset_impl):
        if dataset_impl == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

    def make_all(lang, vocab, dataset_impl=args.dataset_impl):
        if args.trainpref:
            make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers, dataset_impl=dataset_impl)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                # outprefix = "valid{}".format(k) if k > 0 else "valid"

                # NOTE infer the "split" name from prefix, which may include partial paths
                split = Path(validpref).stem
                outprefix = "{}{}".format(split, k) if k > 0 else split

                make_dataset(
                    vocab, validpref, outprefix, lang, num_workers=args.workers, dataset_impl=dataset_impl
                )
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                # outprefix = "test{}".format(k) if k > 0 else "test"

                # NOTE infer the "split" name from prefix, which may include partial paths
                split = Path(testpref).stem
                outprefix = "{}{}".format(split, k) if k > 0 else split

                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers, dataset_impl=dataset_impl)

    def make_all_alignments():
        if args.trainpref and os.path.exists(args.trainpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.trainpref + "." + args.align_suffix,
                "train.align",
                num_workers=args.workers,
            )
        if args.validpref and os.path.exists(args.validpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.validpref + "." + args.align_suffix,
                "valid.align",
                num_workers=args.workers,
            )
        if args.testpref and os.path.exists(args.testpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.testpref + "." + args.align_suffix,
                "test.align",
                num_workers=args.workers,
            )

    # NOTE we both copy the original source tokens and encode the source sentences with dictionary.
    #      For the case when the source embeddings are directly provided
    #      (e.g. from RoBERTa), the source dictionary here and the encoded src values are of no use
    if run_basic:
        make_all(args.source_lang, src_dict, dataset_impl='raw')
        make_all(args.source_lang, src_dict, dataset_impl='mmap')
        # above: just leave for the sake of model to run without too much change
        # NOTE there are <unk> in valid and test set for target actions
        # if target:
        #     make_all(args.target_lang_nopos, tgt_dict)

        # binarize pointer values and save to file

        # TODO make naming convention clearer
        # assume one training file, one validation file, and one test file
        # for pos_file, split in [(f'{pref}.actions_pos', split) for pref, split in
        #                         [(args.trainpref, 'train'), (args.validpref, 'valid'), (args.testpref, 'test')]
        #                         if pref is not None
        #                         ]:
        #     out_pref = os.path.join(args.destdir, split)
        #     task.binarize_actions_pointer_file(pos_file, out_pref)

        # for dynamic oracle: copy the gold graph (with alignments if any) to the data folder
        # TODO to refactor the code here based on need
        if args.task == 'action_pointer_dyo':
            for pref, split in [(args.trainpref, 'train'), (args.validpref, 'valid'), (args.testpref, 'test')]:
                # NOTE infer the "split" name from prefix, which may include partial paths
                split = Path(pref).stem
                split_graph = f'ref_{split}.graph'
                shutil.copyfile(
                    os.path.join(os.path.dirname(pref), split_graph),
                    os.path.join(args.destdir, f'{split}.gold-graph')
                )

        if args.align_suffix:
            make_all_alignments()

    # save actions (output, input, pointers) and states information to assist training with auxiliary info
    # assume one training file, one validation file, and one test file
    if run_act_states:
        task_obj = task(args, tgt_dict=tgt_dict)
        for prefix, split in zip([args.trainpref, args.validpref, args.testpref], ['train', 'valid', 'test']):
            if prefix is None:
                continue
            src_file = prefix + f'.{args.source_lang}'
            actions_file = prefix + '.actions'
            # machine_config_file = os.path.join(os.path.dirname(prefix), 'machine_config.json')
            # NOTE infer the "split" name from prefix, which may include partial paths
            split = Path(prefix).stem
            out_file_pref = os.path.join(args.destdir, split)
            task_obj.build_actions_states_info(src_file, actions_file, out_file_pref,
                                               num_workers=args.workers)
        # create empty file flag
        open(os.path.join(args.destdir, '.done'), 'w').close()

    # save pretrained (e.g. RoBERTa) embeddings
    # TODO refactor this code
    if run_pretrained_feat:
        make_pretrained_features(args, tokenize=tokenize)
        # create empty file flag
        open(os.path.join(args.embdir, '.done'), 'w').close()

    logger.info("Wrote preprocessed src and actions (states) data to {}".format(args.destdir))
    logger.info("Wrote preprocessed pretrained features (embeddings or encodings) data to {}".format(args.embdir))

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        freq_map = {}
        with open(args.alignfile, "r", encoding="utf-8") as align_file:
            with open(src_file_name, "r", encoding="utf-8") as src_file:
                with open(tgt_file_name, "r", encoding="utf-8") as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = src_dict.encode_line(s, add_if_not_exist=False)
                        ti = tgt_dict.encode_line(t, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split("-")), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(
            os.path.join(
                args.destdir,
                "alignment.{}-{}.txt".format(args.source_lang, args.target_lang),
            ),
            "w",
            encoding="utf-8",
        ) as f:
            for k, v in align_dict.items():
                print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=False, tokenize=tokenize_line):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, lang, "bin"),
        impl=args.dataset_impl,
        vocab_size=len(vocab),
        dtype=np.int64,
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(
        filename, vocab, consumer, append_eos=append_eos, offset=offset, end=end, tokenize=tokenize
    )
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def binarize_alignments(args, filename, parse_alignment, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, None, "bin"),
        impl=args.dataset_impl,
        vocab_size=None,
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(
        filename, parse_alignment, consumer, offset=offset, end=end
    )
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    if lang is not None:
        lang_part = ".{}-{}.{}".format(args.source_lang, args.target_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(args.source_lang, args.target_lang)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
