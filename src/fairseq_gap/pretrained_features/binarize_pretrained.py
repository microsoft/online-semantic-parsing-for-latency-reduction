# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
import torch
import shutil
import time
from pathlib import Path

from .sentence_encoding_bart import get_scatter_indices
from ..data import indexed_dataset
from ..utils import time_since


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.embdir, output_prefix)
    lang_part = (
        ".{}-{}.{}".format(args.source_lang, args.target_lang, lang) if lang is not None else ""
    )
    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def make_binary_pretrained_features(args, input_prefix, output_prefix, tokenize):

    # Load pretrained embeddings extractor
    if args.pretrained_embed.startswith('roberta'):
        from .sentence_embedding_roberta import SentenceEmbeddingRoberta

        pretrained_embeddings = SentenceEmbeddingRoberta(
            args.pretrained_embed,
            args.bert_layers,
            remove_be=args.remove_be,
            avg_word=args.avg_word
        )
        no_embeddings = False

    elif args.pretrained_embed.startswith('bert'):
        from .sentence_embedding_bert import SentenceEmbeddingBert

        pretrained_embeddings = SentenceEmbeddingBert(
            args.pretrained_embed,
            args.bert_layers
        )
        no_embeddings = False

    elif args.pretrained_embed.startswith('bart'):
        from .sentence_encoding import SentenceEncodingBART

        # NOTE only encode token ids in BART bpe vocabulary, not the pretrained embedding features
        pretrained_embeddings = SentenceEncodingBART(
            args.pretrained_embed
        )
        no_embeddings = True

    else:
        raise ValueError('arg.pretrained_embed should be either roberta.* or bert-* or bart.*')

    # will store pre-extracted RoBERTa or BERT layer
    if not no_embeddings:
        indexed_data = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, f'{args.source_lang}.embs', "bin"),
            impl=args.dataset_impl,
            dtype=np.float32
        )

    # will store wordpieces and wordpiece to word mapping
    indexed_wordpieces = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, f'{args.source_lang}.wordpieces', "bin"),
        impl=args.dataset_impl,
    )

    indexed_wp2w = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, f'{args.source_lang}.wp2w', "bin"),
        impl=args.dataset_impl,
    )

    num_sents = 0
    input_file = input_prefix + f'.{args.source_lang}'

    start = time.time()
    with open(input_file, 'r') as fid:
        for sentence in fid:

            # we only have tokenized data so we feed whitespace separated
            # tokens
            sentence = " ".join(tokenize(str(sentence).rstrip()))

            # extract embeddings, average them per token and return wordpieces anyway
            if not no_embeddings:
                word_features, wordpieces, word2piece = pretrained_embeddings.extract(sentence)
            else:
                wordpieces, word2piece = pretrained_embeddings.encode_sentence(sentence)

            # note that data needs to be stored as a 1d array. Also check
            # that number nof woprds matches with embedding size
            if not no_embeddings:
                if not args.remove_be and not args.avg_word:
                    assert word_features.shape[1] == len(wordpieces)    # not average to words and keep BOS/EOS
                if args.remove_be and args.avg_word:
                    assert word_features.shape[1] == len(sentence.split())    # average to words and remove BOS/EOS
                indexed_data.add_item(word_features.cpu().view(-1))

            # just store the `wordpieces` indices, including BOS/EOS tokens
            # `word2piece` excluding BOS/EOS tokens
            indexed_wordpieces.add_item(wordpieces)
            indexed_wp2w.add_item(
                get_scatter_indices(word2piece, reverse=True)
            )

            # udpate number of sents
            num_sents += 1
            if not num_sents % 100:
                print("\r%d sentences (time: %s)" % (num_sents, time_since(start)), end='')
        print("")

    # close indexed data files
    if not no_embeddings:
        indexed_data.finalize(
            dataset_dest_file(args, output_prefix, f'{args.source_lang}.embs', "idx")
        )

    indexed_wordpieces.finalize(
        dataset_dest_file(args, output_prefix, f'{args.source_lang}.wordpieces', "idx")
    )
    indexed_wp2w.finalize(
        dataset_dest_file(args, output_prefix, f'{args.source_lang}.wp2w', "idx")
    )

    # copy the source sentence file to go together with the embeddings
    shutil.copyfile(input_file, dataset_dest_prefix(args, output_prefix, args.source_lang))


def make_pretrained_features(args, tokenize=None):
    '''
    Makes BART sentence bpe encodings for source words, or RoBERTa/BERT sentence embeddings
    for the source sentence.
    '''

    assert tokenize

    if args.trainpref:
        make_binary_pretrained_features(args, args.trainpref, "train", tokenize)

    if args.validpref:
        for k, validpref in enumerate(args.validpref.split(",")):
            # outprefix = "valid{}".format(k) if k > 0 else "valid"
            # NOTE infer the "split" name from prefix, which may include partial paths
            split = Path(validpref).stem
            outprefix = "{}{}".format(split, k) if k > 0 else split
            make_binary_pretrained_features(args, validpref, outprefix, tokenize)

    if args.testpref:
        for k, testpref in enumerate(args.testpref.split(",")):
            # outprefix = "test{}".format(k) if k > 0 else "test"
            # NOTE infer the "split" name from prefix, which may include partial paths
            split = Path(testpref).stem
            outprefix = "{}{}".format(split, k) if k > 0 else split
            make_binary_pretrained_features(args, testpref, outprefix, tokenize)
