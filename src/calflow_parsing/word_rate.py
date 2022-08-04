# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterator, List, Tuple, Optional, Dict, Iterable
from pathlib import Path
import numpy as np

from dataflow.core.dialogue import TurnId

from calflow_parsing.conf import ASR_NO_PROGRAMS_DIR, VOICE_ACTED_ASR_DIR

# timings are measured in units of 100 nanoseconds,
# so need to divide by 10,000,000 to get seconds
PER_SEC = 10_000_000

# a fit linear model chars/sec
SLOPE = 0.05502014
INTERCEPT = 0.11375083


@dataclass(frozen=True)
class AsrWord:
    word: str
    offset: int
    duration: int

    @property
    def offset_secs(self) -> float:
        return self.offset / PER_SEC

    @property
    def duration_secs(self) -> float:
        return self.duration / PER_SEC

    @property
    def end(self) -> int:
        return self.offset + self.duration

    @staticmethod
    def from_json(word) -> "AsrWord":
        return AsrWord(
            word=word["Word"],
            offset=word["Offset"],
            duration=word["Duration"],
        )


@dataclass(frozen=True)
class AsrSentence:
    words: List[AsrWord]
    turn_id: Optional[TurnId] = None

    @staticmethod
    def from_json(json) -> "AsrSentence":
        return AsrSentence([
            AsrWord.from_json(word) for word in json
        ])


def simulate_asr_timings(tokens: List[str], slope: float = SLOPE, intercept: float = INTERCEPT) -> Iterator[AsrWord]:
    offset = 0
    for token in tokens:
        duration = int(PER_SEC * (len(token) * slope + intercept))
        yield AsrWord(word=token, offset=offset, duration=duration)
        offset = offset + duration


def load_fdr_json(asr_dir: Path = ASR_NO_PROGRAMS_DIR) -> Iterator[AsrSentence]:
    paths = asr_dir.glob("*.log")
    for path in paths:
        with open(path, encoding="utf-8") as file:
            for result in json.load(file)['results']:
                if 'NBest' in result:
                    # only care about 1-best for now
                    best = result['NBest'][0]
                    if 'Words' in best:
                        yield AsrSentence.from_json(best['Words'])


def load_voice_acted_json(asr_dir: Path = VOICE_ACTED_ASR_DIR) -> Iterator[AsrSentence]:
    paths = asr_dir.glob("*.jsonl")
    for path in paths:
        print('-' * 10 + f' Loading voice acted ASR results from {path} ' + '-' * 10)
        with open(path, encoding="utf-8") as file:
            for line in file:
                result = json.loads(line)
                if 'NBest' in result:
                    # only care about 1-best for now
                    best = result['NBest'][0]
                    if 'Words' in best:
                        sentence = AsrSentence.from_json(best['Words'])
                        dialogue_id, turn_idx = result['utterance_id'].split('_')
                        turn_id = TurnId(dialogue_id=dialogue_id, turn_index=int(turn_idx))
                        yield replace(sentence, turn_id=turn_id)


def collate_voice_acted_json(sentences: Iterable[AsrSentence]) -> Iterator[AsrSentence]:
    """Some turns got split apart by ASR. Here we put them back together."""
    # add a tenth of a second between splices
    splice_pause = int(0.1 * PER_SEC)
    by_turn_id: Dict[TurnId, List[AsrSentence]] = defaultdict(list)
    for s in sentences:
        by_turn_id[s.turn_id].append(s)
    for turn_id, ss in by_turn_id.items():
        if ss:
            offset = ss[0].words[0].offset
            words = []
            for s in ss:
                if s.words:
                    s_offset = s.words[0].offset
                    diff = offset - s_offset
                    words.extend([
                        replace(w, offset=w.offset + diff)
                        for w in s.words
                    ])
                    offset = s.words[-1].end + splice_pause
            yield AsrSentence(words=words, turn_id=ss[0].turn_id)


def durations(words: List[AsrWord], start: Optional[int] = None, remove_outlier=True) -> Iterator[Tuple[str, int]]:
    """Includes the pause before"""
    if words:
        first, *rest = words
        start = first.offset if start is None else start
        end = first.offset + first.duration
        if remove_outlier:
            if end - start > 2 * PER_SEC:
                print(first.word, (end - start) / PER_SEC, first.duration_secs)
            else:
                yield first.word, end - start
        else:
            yield first.word, end - start
        yield from durations(rest, start=end, remove_outlier=remove_outlier)


def word_rate_mean_and_std(sentences: Iterator[AsrSentence]) -> Tuple[float, float]:
    ds = [
        d / PER_SEC
        for s in sentences
        for _, d in durations(s.words)
    ]
    return np.mean(ds), np.std(ds)


def chars_to_seconds_linear_model(sentences: Iterator[AsrSentence]) -> Tuple[float, float, float]:
    """Returns slope and intercept"""
    all_durations = [
        (w, d / PER_SEC)
        for s in sentences
        for w, d in durations(s.words)
    ]
    chars = np.array([len(w) for w, _ in all_durations])
    # add bias feature
    A = np.vstack([chars, np.ones(len(chars))]).T
    b = np.array([s for _, s in all_durations])
    # fit least-squares linear model
    lstsq = np.linalg.lstsq(A, b, rcond=None)
    (slope, intercept), (squared_error,), *_ = lstsq
    return slope, intercept, np.sqrt(squared_error / len(all_durations))
