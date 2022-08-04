# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import os

from fairseq.tasks import register_task

from .action_pointer_scp import ActionPointerSrcCopyParsingTask
from fairseq_gap.actions_interface.source_copy import CalFlowActionsSrcCopyInterface


logger = logging.getLogger(__name__)


@ register_task('action_pointer_scp_prefix')
class ActionPointerSrcCopyPrefixParsingTask(ActionPointerSrcCopyParsingTask):

    @classmethod
    def preprocess_action_files(cls, actions_filenames, source_filenames):
        """Preprocess action files so that the resulted action files can be used for tgt dictionary construction.
        Here the preprocess includes:
        - from both the source tokens and actions, get the copy actions with src pointers, save the file with
        '.actions.src_copy'.
        - strip all the pointer values, including tgt edge pointers and src copy pointers, to have '.actions_nopos'
        for dictionary building; and pointers are saved at '.actions_pos' and '.actions_src_pos'.

        Args:
            actions_filenames ([type]): [description]
        """
        for src_filename, actions_filename in zip(source_filenames, actions_filenames):
            actions_filename = os.path.abspath(actions_filename)
            actions_ext = os.path.splitext(actions_filename)[1]
            assert actions_ext == '.actions', 'graph actions file name must end with ".actions"'

            # we already start with the actions with src copy actions

            # get the actions file with no pointer (peeling off tgt edge pointer and src copy pointer)
            # resulted files are with postfix '.actions_nopos', '.actions_pos', '.actions_src_pos'
            cls.split_actions_pointer(actions_filename + '.src_copy')

            # recover the original actions that can be run with the base state machine
            CalFlowActionsSrcCopyInterface.recover_copy_with_token_file(src_filename,
                                                                        actions_filename + '.src_copy',
                                                                        actions_filename,
                                                                        tokenize=cls.tokenize)
