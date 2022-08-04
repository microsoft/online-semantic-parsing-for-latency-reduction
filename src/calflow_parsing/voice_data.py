# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
from dataclasses import replace
from typing import List, Tuple
import json

import jsons
from tqdm import tqdm
from nltk import edit_distance

from dataflow.core.dialogue import Dialogue, TurnId
from calflow_parsing.word_rate import load_voice_acted_json, durations, AsrWord, AsrSentence
from calflow_parsing.conf import DATA_DIR, SAVE_DIR, VOICE_ACTED_ASR_DIR
from calflow_parsing.dtw import dtw


VOICE_DATA_PATH = DATA_DIR / 'smcalflow2.0_voice300' / 'dev_100.modified.jsonl'
# VOICE_DATA_PATH = DATA_DIR / 'smcalflow2.0_voice300' / 'test_100.merged.modified.jsonl'

ORIGINAL_DIALOGUE_PATH = DATA_DIR / 'smcalflow2.0' / 'valid.dataflow_dialogues.jsonl'
PROCESSED_SRC_PATH = (DATA_DIR / 'processed' / 'smcalflow2.0'
                      / 'act-top_src-ct1-npwa_tgt-cp-str' / 'oracle' / 'valid.utters')


def get_utterance_id(dataflow_dialogues_jsonl: str) -> List[TurnId]:
    """Get the utterance_id for each example in the dataset.

    Args:
        dataflow_dialogues_jsonl (str): original data path.
    """
    print('-' * 10 + f' Loading original validation dialogue data from {dataflow_dialogues_jsonl} ' + '-' * 10)

    all_turn_ids = []

    for line in tqdm(open(dataflow_dialogues_jsonl), unit=" dialogues"):
        dialogue: Dialogue
        dialogue = jsons.loads(line.strip(), Dialogue)
        dialogue_id = dialogue.dialogue_id

        for turn in dialogue.turns:
            if turn.skip:
                continue

            turn_index = turn.turn_index
            all_turn_ids.append(TurnId(dialogue_id, turn_index))

    return all_turn_ids


def get_utterance_from_src(processed_utterance_src: str) -> List[List[str]]:
    """Get the utterance from processed src file with possible contexts and special marks.

    Args:
        processed_utterance_src (str): processed utterance src file path.
    """
    print('-' * 10 + f' Loading processed validation utterances data from {processed_utterance_src} ' + '-' * 10)

    all_tokenized_utterances: List[List[str]] = []
    with open(processed_utterance_src, 'r') as f:
        for line in tqdm(f, unit=' utterances'):
            if line.strip():
                context = line.strip().split('__User')[:-1]
                utterance = line.strip().split('__User')[-1]
                utterance = utterance.split()[:-1]    # remove the last '__StartOfProgram' mark
                all_tokenized_utterances.append(utterance)

    return all_tokenized_utterances


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voice-data-dir', type=str,
                        default=VOICE_ACTED_ASR_DIR,
                        help='dir containing the voice acted ASR results with utterance timing')
    parser.add_argument('--dialogues-jsonl', type=str,
                        default=ORIGINAL_DIALOGUE_PATH,
                        help='the jsonl file containing the dialogue data with dataflow programs')
    parser.add_argument('--source-utterances', type=str,
                        default=PROCESSED_SRC_PATH,
                        help='file path of the tokenized source utterances (with contexts)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # ===== load in ASR 300 voice data
    asr_sentences = list(load_voice_acted_json(asr_dir=args.voice_data_dir))

    # NOTE filter out the problematic examples, e.g.
    # asr_sentences[128]:
    # AsrSentence(words=[
    #   AsrWord(word='should', offset=16800000, duration=17900000),
    #   AsrWord(word='should', offset=16800000, duration=17900000),
    #   AsrWord(word='definitely', offset=20000000, duration=5700000),
    #   AsrWord(word='be', offset=25800000, duration=1800000),
    #   AsrWord(word='earlier', offset=27700000, duration=7000000)],
    # turn_id=TurnId(dialogue_id='26c5663d-81c2-49da-9f6d-03ceb540e914', turn_index=3))
    #
    # The above could cause negative durations!
    for iidx, asr_sent in tqdm(enumerate(asr_sentences), unit=' asr_sentences', desc='filtering out problematic asr'):
        # remove the repetitive asr words
        new_asr_words = []
        aw_last = None
        for aw in asr_sent.words:
            if aw != aw_last:
                new_asr_words.append(aw)
            aw_last = aw
        # update the duration so that there is no overlap with the next word, which causes negative duration
        # (not the most efficient to separate out the checks, but we have small data)
        for i, aw in enumerate(new_asr_words[:-1]):
            if aw.offset + aw.duration > new_asr_words[i + 1].offset:
                # issue: there is an overlap between asr words, this might cause negative duration
                adjusted_duration = new_asr_words[i + 1].offset - aw.offset
                assert adjusted_duration >= 0
                new_asr_words[i] = replace(aw, duration=adjusted_duration)

        # update the asr sentence
        asr_sentences[iidx] = replace(asr_sentences[iidx], words=new_asr_words)

    # ===== load in original validation set data
    all_turn_ids = get_utterance_id(dataflow_dialogues_jsonl=args.dialogues_jsonl)
    all_tokenized_utterances = get_utterance_from_src(processed_utterance_src=args.source_utterances)
    assert len(all_turn_ids) == len(all_tokenized_utterances)

    # ===== map ASR 300 voice data back to the original data, and get the durations for each ASR words
    asr_turn_id_to_index: List[int] = []    # store indexes in the full validation data for the 300 voice data
    asr_words_and_durations: List[List[Tuple[str, int]]] = []
    asr_original_words: List[str] = []    # original tokenized utterances before ASR

    for asr_sent in tqdm(asr_sentences, unit=' asr_sentences'):
        asr_turn_id_to_index.append(all_turn_ids.index(asr_sent.turn_id))
        asr_words_and_durations.append(list(durations(asr_sent.words, remove_outlier=False)))
        asr_original_words.append(all_tokenized_utterances[asr_turn_id_to_index[-1]])

        # # checking ignored ASR word durations due to a long pause
        # if len(asr_sent.words) != len(asr_words_and_durations[-1]):
        #     print(asr_sent.words)
        #     print(asr_words_and_durations[-1])
        #     print(asr_original_words[-1])
        #     breakpoint()

    # breakpoint()

    # ===== run DTW algorithm to find the optimal match between the original tokenized utterance and the ASR results
    print('-' * 10 + ' Matching the original words with the ASR words ' + '-' * 10)
    mapped_asr_sentences = {}
    mapped_asr_sentences_json = {}

    dist_func = lambda x, y: edit_distance(x.lower(), y.lower())    # remove capitalization when comparing
    for iidx, (original_index, original_words, asr_words) in tqdm(
            enumerate(zip(asr_turn_id_to_index, asr_original_words, asr_words_and_durations)),
            unit=' utterances', desc='dtw token and duration matching'):
        asr_words, asr_durations = zip(*asr_words)
        dtw_dist, path = dtw(original_words, asr_words, dist=dist_func)

        # === construct the AsrSentence with timing for the original utterance based on the mapping
        mapped_asr_words = []

        offset = 0
        end = 0
        path_idx = 0
        pre_asr_idx = -1    # denote the last added ASR duration (to not accumulate twice)
        for i, word in enumerate(original_words):
            # find the last aligned asr_word
            assert i == path[path_idx][0], 'every original word should be matched to asr words'

            # exit condition: currently the `path_idx` is at the next original word,
            # or we have enumerated all matched tuples in `path`
            while path_idx < len(path) and path[path_idx][0] == i:
                cur_asr_idx = path[path_idx][1]
                # NOTE only update `end` when we reach to a new ASR word (do not repeatedly accumulate the duration)
                if cur_asr_idx != pre_asr_idx:
                    end += asr_durations[cur_asr_idx]
                    pre_asr_idx = cur_asr_idx
                path_idx += 1

            # get the duration and offset, create AsrWord
            duration = end - offset
            asr_word = AsrWord(word=word, offset=offset, duration=duration)
            mapped_asr_words.append(asr_word)

            # # debug: for the issue that the duration is negative
            # # if duration < 0:
            # if iidx in [128, 172, 216, 294]:    # problematic examples
            #     print()
            #     print('original:', original_words)
            #     print('asr:', asr_words)
            #     print('dtw_distance:', dtw_dist)
            #     print('matched_path:', path)
            #     print('asr_word:', asr_word)
            #     print('asr_words_and_durations:', asr_words_and_durations[iidx])
            #     print('idx:', iidx)
            #     breakpoint()

            offset = end

        # create AsrSentence
        mapped_asr_sentences[original_index] = AsrSentence(words=mapped_asr_words)
        mapped_asr_sentences_json[original_index] = [{'Word': aw.word, 'Offset': aw.offset, 'Duration': aw.duration}
                                                     for aw in mapped_asr_words]

        # # debug and print
        # print()
        # print('original:', original_words)
        # print('asr:', asr_words)
        # print('dtw_distance:', dtw_dist)
        # print('matched_path:', path)
        # print('mapped_asr_sentence:', mapped_asr_sentences[original_index])
        # breakpoint()

    # breakpoint()

    # ===== store the processed words and duration into a file
    json.dump(mapped_asr_sentences_json, open(VOICE_ACTED_ASR_DIR / 'mapped_val_300.jsonl', 'w'))
    print('-' * 10 + ' Mapped duration from voice acting to original 300 validation utterances, '
          f'saved at {VOICE_ACTED_ASR_DIR / "mapped_val_300.jsonl"}')
