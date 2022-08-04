# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import os

from fairseq.tasks import FairseqTask, register_task

from fairseq_gap.actions_interface.source_copy import (
    CalFlowActionsSrcCopyInterface,
    peel_edge_and_copy_pointer
    )
from .action_pointer import ActionPointerParsingTask, load_action_pointer_dataset
from fairseq_gap.actions_preprocess.action_data_binarize_scp import (
    ActionStatesBinarizer,
    binarize_actstates_tofile_workers,
    load_actstates_fromfile
)


logger = logging.getLogger(__name__)


@ register_task('action_pointer_scp')
class ActionPointerSrcCopyParsingTask(ActionPointerParsingTask):

    @classmethod
    def split_actions_pointer(cls, actions_filename):
        """Split the actions file into 3 files, one with no pointers at all (used for building the dictionary),
        one with target edge pointers, and one with source copy pointers.

        Args:
            actions_filename ([type]): [description]
        """
        actions_filename = os.path.abspath(actions_filename)
        assert actions_filename.endswith('.actions.src_copy'), \
            'graph actions (with source copy action) file name must end with ".actions.src_copy"'
        actions_outfile_pref = os.path.splitext(actions_filename)[0]    # removes the '.src_copy' part

        with open(actions_filename, 'r') as f, \
                open(actions_outfile_pref + '_nopos', 'w') as g, open(actions_outfile_pref + '_pos', 'w') as h, \
                    open(actions_outfile_pref + '_src_pos', 'w') as h2:
            for line in f:
                if not line.strip():
                    continue
                line_actions = cls.tokenize(line)
                line_actions = [peel_edge_and_copy_pointer(act) for act in line_actions]
                line_actions_nopos, line_actions_pos, line_actions_src_pos = zip(*line_actions)
                g.write(cls.word_sep.join(line_actions_nopos))
                g.write('\n')
                h.write(cls.word_sep.join(map(str, line_actions_pos)))
                h.write('\n')
                h2.write(cls.word_sep.join(map(str, line_actions_src_pos)))
                h2.write('\n')

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

            # generate the actions file with src copy
            # # reset the class config if provided
            # CalFlowActionsSrcCopyInterface.reset_config(copy_only_string=copy_only_string)
            num_avg_copies, num_total_copies, num_lines = \
                CalFlowActionsSrcCopyInterface.reform_actions_with_copy_file(
                    src_filename,
                    actions_filename,
                    out_file_pref=None,
                    only_string=None,    # use the default from class attribute
                    only_from_current_utter=True,
                    tokenize=cls.tokenize,
                    pad=-1
                    )
            # this would generate files 'actions.src_copy', 'actions.src_pos'
            logger.info(f'[actions with src copy] {actions_filename + ".src_copy"}: ')
            logger.info(f'Number of average copies per action sequence: '
                        f'{num_avg_copies:.2f} ({num_total_copies} / {num_lines})')

            # get the actions file with no pointer (peeling off tgt edge pointer and src copy pointer)
            # resulted files are with postfix '.actions_nopos', '.actions_pos', '.actions_src_pos'
            cls.split_actions_pointer(actions_filename + '.src_copy')

    def build_actions_states_info(self, en_file, actions_file, out_file_pref, num_workers=1):
        """Preprocess to get the actions states information and save to binary files.

        Args:
            en_file (str): English sentence file path.
            actions_file (str): actions file path, with copy actions.
            out_file_pref (str): output file prefix.
            num_workers (int, optional): number of workers for multiprocessing. Defaults to 1.
        """
        # make sure the actions file is the one after preprocessing to have copy actions
        assert os.path.splitext(actions_file)[1] == '.actions'
        actions_file_scp = actions_file + '.src_copy'

        out_file_pref = out_file_pref + f'.{self.args.source_lang}-actions.actions'
        if self.action_state_binarizer is None:
            # for reuse (e.g. train/valid/test data preprocessing) to avoid building the canonical action to
            # dictionary id mapping repeatedly
            self.action_state_binarizer = ActionStatesBinarizer(self.tgt_dict)
        res = binarize_actstates_tofile_workers(en_file, actions_file_scp, out_file_pref,
                                                action_state_binarizer=self.action_state_binarizer,
                                                impl='mmap', tokenize=self.tokenize, num_workers=num_workers)
        logger.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                'actions',
                actions_file + '_nopos',
                res['nseq'],
                res['ntok'],
                100 * res['nunk'] / (res['ntok'] + 1e-6),    # when it is not recorded: denominator being 0
                self.tgt_dict.unk_word,
            )
        )

    def load_dataset(self, split, epoch=0, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split.startswith('valid'):    # could be "valid-prefix50p", etc.
            paths = self.args.data_valid.split(':')
            emb_dir = self.args.emb_dir_valid
        else:
            paths = self.args.data.split(':')
            emb_dir = self.args.emb_dir
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_action_pointer_dataset(
            data_path, emb_dir, split,
            src, tgt, self.src_dict, self.tgt_dict,
            self.tokenize,
            dataset_impl=self.args.dataset_impl,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            shuffle=True,
            append_eos_to_target=self.args.append_eos_to_target,
            collate_tgt_states=self.args.collate_tgt_states,
            collate_tgt_states_graph=self.args.collate_tgt_states_graph,
            src_fix_emb_use=self.args.src_fix_emb_use,
            src_fix_emb_dim=getattr(self.args, 'src_pretrained_emb_dim', None) or self.args.src_fix_emb_dim,
            load_actstates_fromfile=load_actstates_fromfile,
        )

    def build_generator(self, args, model_args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq_gap.sequence_generator_scp import SequenceGenerator
            return SequenceGenerator(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                stop_early=(not getattr(args, 'no_early_stop', False)),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                shift_pointer_value=getattr(model_args.criterion, 'shift_pointer_value', 0),
                stats_rules=getattr(args, 'machine_rules', None),
                machine_config_file=getattr(args, 'machine_config', None)
            )
