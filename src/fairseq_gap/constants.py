# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from calflow_parsing.calflow_machine import (
    RA,
    LA,
    STRING_MARK,
    STRING_MARK_END,
    join_action_pointer
    )


class SpecialSymbols:
    # arc/edge symbol in actions
    RA = RA
    LA = LA
    STRING_MARK = STRING_MARK
    STRING_MARK_END = STRING_MARK_END
    COPY = '_COPY_'    # NOTE in src utterances, there are 'COPY'
    COPY_ANONYM = '_CPANY_'


join_action_pointer = join_action_pointer
