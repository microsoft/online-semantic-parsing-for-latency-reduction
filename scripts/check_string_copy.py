# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Check string copies in the target lispress compared to the source utterance.
Tokenization of the strings matters.
"""
import sys
from typing import Dict
from collections import Counter
from itertools import chain

import jsons
from tqdm import tqdm

from dataflow.core.dialogue import Dialogue, Turn, TurnId
from dataflow.core.program import ValueOp


def extract_strings(program):
    strings = []
    for exp in program.expressions:
        if isinstance(exp.op, ValueOp):
            value = jsons.loads(exp.op.value)
            schema = value['schema']
            underlying = value['underlying']

            if schema == 'String':
                strings.append(underlying)

    return strings


def check_copy(strings, utter_tokens, tokenize):
    uncopied = []
    for string in strings:
        for tok in tokenize(string):
            if tok in utter_tokens:
                pass
            else:
                uncopied.append(tok)

    return uncopied


def check_detokenize(strings, tokenize, detokenize):
    unmatched = []
    for string in strings:
        string_tok = tokenize(string)
        string_rec = detokenize(string_tok)
        if string_rec != string:
            unmatched.append((string, string_tok, string_rec))
    return unmatched


def tokenize_split(string):
    return string.split()


def detokenize_split(tokens):
    return ' '.join(tokens)


def tokenize_nltk_wordpunct(string):
    from nltk import wordpunct_tokenize
    return wordpunct_tokenize(string)


def tokenize_nltk_word(string):
    from nltk import word_tokenize
    return word_tokenize(string)


def tokenize_moses(string):
    from sacremoses import MosesTokenizer
    mt = MosesTokenizer(lang='en')
    return mt.tokenize(string, aggressive_dash_splits=True, escape=False, return_str=False)


def detokenize_moses(tokens):
    from sacremoses import MosesDetokenizer
    md = MosesDetokenizer(lang='en')
    string = md.detokenize(tokens, return_str=True, unescape=True)
    return string


def tokenize_gpt2re(string):
    import regex as re
    pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    tokens = re.findall(pat, string)
    tokens = [x.strip() for x in tokens]    # remove the space in the front
    return tokens


def tokenize_gpt2(string):
    import regex as re
    pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    tokens = re.findall(pat, string)
    # add a space after the double quote
    for i, tok in enumerate(tokens):
        if tok == ' "' and i < len(tokens) - 1:
            tokens[i + 1] = ' ' + tokens[i + 1].lstrip()

    return tokens


def detokenize_gpt2(tokens):
    return ''.join(tokens)


def tokenize_simple(string):
    import regex as re
    pat = re.compile(
            r"""\p{N}+|[",;:.?!\-]|[^\s\p{N}",;:.?!\-']+|'\S*"""
        )
    tokens = re.findall(pat, string)
    return tokens


def detokenize_simple(tokens):
    import regex as re
    pat_number = re.compile(r"""\p{N}+""")
    pat_prime = re.compile(r"""'\S*""")
    pat_quote = re.compile(r"""["]""")
    pat_dash = re.compile(r"""[-]""")
    pat_punct = re.compile(r"""[,;:.?!]""")
    pat_special = re.compile(r"""(D . C .)|(D . C)|(P . F .)|(J . J .)|(CS : GO)|(D . R .)|(J . P .)|(: D)|(D . J)|(D . J .)""")

    tok_string = ' '.join(tokens)
    # ===== dealing with numbers
    string = ''
    end = 0
    for m in re.finditer(pat_number, tok_string):

        if tok_string[m.start() - 2:m.start()] in [': ', '# ', '$ ', '/ '] \
            or \
                (tok_string[m.start() - 2:m.start() - 1].isupper()
                 and tok_string[m.start() - 4:m.start() - 1] not in ['PHO']) \
                     or \
                         tok_string[m.start():m.end()] == '000':
            # time, e.g. 3 : 30
            string += tok_string[end:m.start() - 1]
        else:
            string += tok_string[end:m.start()]

        string += tok_string[m.start():m.end()]

        if tok_string[m.end():m.end() + 1] == ' ':
            # this is not always true, e.g. "Joey's27thBirthday" is tokenized as ['Joey', "'s27thBirthday"]
            # this seems to be just bad data
            if (m.end() == len(tok_string) - 3 and tok_string[m.end() + 1:m.end() + 3] in ['th', 'st', 'nd', 'rd']) \
                or tok_string[m.end() + 1:m.end() + 4] in ['th ', 'st ', 'nd ', 'rd ', 'ers', 'ER\''] \
                    or (m.end() == len(tok_string) - 2 and tok_string[m.end()+1:m.end() + 2] in ['k', 'K', 'm', 'M', 'A', 'B', 'a', 'b', 'D', '$', '%']) \
                        or tok_string[m.end() + 1:m.end() + 3]  in ['k ', 'K ', 'm ', 'M ', '/ ']:
                end = m.end() + 1
            else:
                end = m.end()
        else:
            end = m.end()

    # last segment
    string += tok_string[end:]

    # ===== dealing with prime
    tok_string = string
    string = ''
    end = 0
    for m in re.finditer(pat_prime, tok_string):

        assert tok_string[m.start() - 1] == ' '
        string += tok_string[end:m.start() - 1]

        string += tok_string[m.start():m.end()]

        end = m.end()
    # last segment
    string += tok_string[end:]

    # ===== dealing with quotes
    tok_string = string
    string = ''
    end = 0
    for i, m in enumerate(re.finditer(pat_quote, tok_string)):

        if not i % 2:
            # left quote
            if m.end() <= len(tok_string) - 1:
                # the condition is not always satisfied, just bad data
                assert tok_string[m.end()] == ' '

            string += tok_string[end:m.end()]
            end = m.end() + 1
        else:
            # right quote
            assert tok_string[m.start() - 1] == ' '
            string += tok_string[end:m.start() - 1]
            string += tok_string[m.start():m.end()]
            end = m.end()
    # last segment
    string += tok_string[end:]

    # ===== dealing with dash
    tok_string = string
    string = ''
    end = 0
    for m in re.finditer(pat_dash, tok_string):
        if tok_string[m.start() - 1] == ' ':
            string += tok_string[end:m.start() - 1]
        else:
            string += tok_string[end:m.start()]
        string += tok_string[m.start():m.end()]
        if tok_string[m.end()] == ' ':
            end = m.end() + 1
        else:
            end = m.end()
    # last segment
    string += tok_string[end:]

    # ===== dealing with special words
    tok_string = string
    string = ''
    end = 0
    for m in re.finditer(pat_special, tok_string):
        string += tok_string[end:m.start()]
        string += ''.join(tok_string[m.start():m.end()].split())
        end = m.end()
    # last segment
    string += tok_string[end:]

    # ===== dealing with other punctuations
    tok_string = string
    string = ''
    end = 0
    for m in re.finditer(pat_punct, tok_string):
        if tok_string[m.start() - 1] == ' ' and tok_string[m.start():m.end() + 1] != ':D':
            string += tok_string[end:m.start() - 1]
        else:
            string += tok_string[end:m.start()]
        string += tok_string[m.start():m.end()]
        end = m.end()
    # last segment
    string += tok_string[end:]

    return string


if __name__ == '__main__':
    dataflow_dialogues_jsonl = sys.argv[1]

    utter_token_counter = Counter()

    uncopied_all = []
    unmatched_all = []

    unmatched_counter = Counter()

    for line in tqdm(open(dataflow_dialogues_jsonl), unit=" dialogues"):
        dialogue: Dialogue
        dialogue = jsons.loads(line.strip(), Dialogue)

        turn_lookup: Dict[int, Turn] = {turn.turn_index: turn for turn in dialogue.turns}
        for turn_index, turn in turn_lookup.items():
            if turn.skip:
                continue

            strings = extract_strings(turn.program())

            utter_string = turn.user_utterance.original_text
            utter_tokens = turn.user_utterance.tokens

            tokenize = tokenize_split
            detokenize = detokenize_split

            # tokenize = tokenize_nltk_wordpunct
            # tokenize = tokenize_nltk_word

            # tokenize = tokenize_moses
            # detokenize = detokenize_moses

            # tokenize = tokenize_gpt2re

            # tokenize = tokenize_gpt2
            # detokenize = detokenize_gpt2
            # # for this, we prepend a space for both src and tgt
            # utter_string = ' ' + utter_string.lstrip()
            # strings = [' ' + string for string in strings]

            tokenize = tokenize_simple
            detokenize = detokenize_simple

            # also tokenize the source with the same tokenizer
            utter_tokens = tokenize(utter_string)
            utter_token_counter.update(utter_tokens)

            uncopied = check_copy(strings, utter_tokens, tokenize)
            uncopied_all.append(uncopied)

            # if uncopied:
            #     while 'output' in uncopied:
            #         uncopied.remove('output')
            #     while 'start' in uncopied:
            #         uncopied.remove('start')
            #     while 'place' in uncopied:
            #         uncopied.remove('place')
            #     while 'time' in uncopied:
            #         uncopied.remove('time')
            #     while 'end' in uncopied:
            #         uncopied.remove('end')
            #     while ' output' in uncopied:
            #         uncopied.remove(' output')
            #     while ' start' in uncopied:
            #         uncopied.remove(' start')
            #     while ' place' in uncopied:
            #         uncopied.remove(' place')
            #     while ' time' in uncopied:
            #         uncopied.remove(' time')
            #     while ' end' in uncopied:
            #         uncopied.remove(' end')
            #     if uncopied:
            #         print('\n')
            #         print(uncopied)
            #         print(utter_string)
            #         print(utter_tokens)
            #         breakpoint()

            unmatched = check_detokenize(strings, tokenize, detokenize)
            unmatched_all.append(unmatched)

            unmatched_counter.update([(x[0], ' '.join(x[1]), x[2]) for x in unmatched])

            # if unmatched:
            #     print('\n')
            #     print(unmatched)
            #     breakpoint()

    print(f'Number of different tokens in src: {len(utter_token_counter)}')

    uncopied_counter = Counter(chain.from_iterable(uncopied_all))
    print(f'Number of uncopied tokens: {len(uncopied_counter)}')
    breakpoint()
    print(uncopied_counter)

    print(f'Number of lispress containing unrecovered strings: {sum(1 if x else 0 for x in unmatched_all)}')
    # breakpoint()
    # print(list(filter(lambda x: x, unmatched_all)))
    print(f'Number of different unrecovered target strings: {len(unmatched_counter)}')
    breakpoint()
    print(unmatched_counter)
