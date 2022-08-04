# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import regex as re


def normalize_equote(string):
    """Normalize text to make sure that '\\"' has a space both before and after, which would make the tokenization
    work as normally (otherwise there might be unexpected behaviors.
    """
    pat_equote = re.compile(r"""(\\")""")
    tok_string = ''
    end = 0
    for m in re.finditer(pat_equote, string):
        if m.start() >= 1:
            if string[m.start() - 1] == ' ':
                tok_string += string[end:m.end()]
            else:
                tok_string += string[end:m.start() - 1] + ' ' + string[m.start():m.end()]
        else:
            tok_string += string[end:m.end()]

        if string[m.end():m.end() + 1] == ' ':
            end = m.end()
        else:
            end = m.end() + 1
    # last segment
    tok_string += string[end:]

    return tok_string


def tokenize(string):
    string = normalize_equote(string)
    pat = re.compile(
            r"""\\"|\p{N}+|[",;:.?!\-]|[^\s\p{N}",;:.?!\-']+|'\S*"""
        )
    # NOTE in the lispress strings there are examples like "\"Get Well Soon!\" card for Jane"
    tokens = re.findall(pat, string)
    return tokens


def detokenize(tokens):
    import regex as re
    pat_number = re.compile(r"""\p{N}+""")
    pat_prime = re.compile(r"""'\S*""")
    pat_quote = re.compile(r"""["]""")
    pat_equote = re.compile(r"""(\\")""")
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
                        or tok_string[m.end() + 1:m.end() + 3] in ['k ', 'K ', 'm ', 'M ', '/ ']:
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

        if m.start() >= 1:
            # assert tok_string[m.start() - 1] == ' '
            # NOTE we comment the above because of corner case: from DLM completion there is an example of "Jared's" generation in
            # ../DATA/processed/smcalflow2.0/act-top_src-ct1-npwa_tgt-cp-str_from-utter-dlm-beam5_abs-prefix-all/oracle/valid-prefixallt-beam5-2.utters
            # for which the parsing results copied over to the string, which violate the above assert condition.
            string += tok_string[end:m.start() - 1]

        string += tok_string[m.start():m.end()]

        end = m.end()
    # last segment
    string += tok_string[end:]

    # ===== dealing with quotes
    tok_string = string
    string = ''
    end = 0
    i = 0    # counter for the quote
    for m in re.finditer(pat_quote, tok_string):

        # NOTE do not look at the literal quote here
        if tok_string[m.start() - 1:m.start() + 1] == '\\"':
            continue

        if not i % 2:
            # left quote
            if m.end() <= len(tok_string) - 1:
                # the condition is not always satisfied, just bad data or bad generation (e.g. "'s)
                if tok_string[m.end()] == ' ':
                    end = m.end() + 1
                else:
                    end = m.end()
            else:
                end = m.end()

            string += tok_string[end:m.end()]
        else:
            # right quote
            if tok_string[m.start() - 1] == ' ':
                string += tok_string[end:m.start() - 1]
            else:
                string += tok_string[end:m.start()]

            string += tok_string[m.start():m.end()]
            end = m.end()

        i += 1
    # last segment
    string += tok_string[end:]

    # ===== dealing with literal escaped quotes '\\"'
    tok_string = string
    string = ''
    end = 0
    for i, m in enumerate(re.finditer(pat_equote, tok_string)):

        if not i % 2:
            # left quote
            if m.end() <= len(tok_string) - 1:
                # the condition is not always satisfied, just bad data or bad generation (e.g. \"'s)
                if tok_string[m.end()] == ' ':
                    end = m.end() + 1
                else:
                    end = m.end()
            else:
                end = m.end()

            string += tok_string[end:m.end()]
        else:
            # right quote
            if tok_string[m.start() - 1] == ' ':
                string += tok_string[end:m.start() - 1]
            else:
                string += tok_string[end:m.start()]

            string += tok_string[m.start():m.end()]
            end = m.end()
    # last segment
    string += tok_string[end:]

    # ===== dealing with dash
    tok_string = string
    string = ''
    end = 0
    for m in re.finditer(pat_dash, tok_string):
        if m.start() >= 1 and tok_string[m.start() - 1] == ' ':
            string += tok_string[end:m.start() - 1]
        else:
            string += tok_string[end:m.start()]
        string += tok_string[m.start():m.end()]
        if tok_string[m.end():m.end() + 1] == ' ':
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


def detokenize_treedst(tokens):
    """De-tokenize TreeDST strings. There are a lot of special strings such as "object.dateTimeRange.startDateTime.time"
    "object.returnDateTime", etc.
    """
    string = detokenize(tokens)

    if string.startswith('object.'):
        string = ''.join(string.split())    # remove all space

    if string.startswith('focus.'):
        string = ''.join(string.split())    # remove all space

    if 'dateTimeRange' in string:
        string = ''.join(string.split())    # remove all space

    # if string[0].isnumeric() and string[1:3] == ' a':
    #     string = string[0] + string[2:]    # remove the space in '2 a', '4 a', '6 a'

    if string[:3] in ['2 a', '4 a']:
        string = string[0] + string[2:]     # remove the space in '2 a'

    import regex as re
    pat_special = re.compile(r"""(\\u 2019 s)|(6 a)""")

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

    return string
