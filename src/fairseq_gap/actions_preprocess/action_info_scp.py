# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys

from tqdm import tqdm

from fairseq_gap.actions_interface.source_copy import CalFlowActionsSrcCopyInterface as InterfaceMachine


def get_actions_states(*, tokens=None, actions=None, append_close=True):
    """Get the information along with runing the graph actions interface state machine with the provided
    action sequence, to get all the actions and states data to be used by the model.

    The information includes:
        - target side output actions (without pointer), input actions, and edge pointer values (associated with output)
        - the allowed actions (canonical form) for each action in the actions sequence ('CLOSE' position can be included)
        - a mask on previous actions to indicate node generation
        - token cursor before each action (for transition-based parsing based on alignments)

    Args:
        tokens (List[str], optional): source token sequences (a sentence). Defaults to None.
        actions (List[str], optional): the action sequence. Defaults to None.
        append_close (bool): whether to include the 'CLOSE' action at the end of the action sequence

    Returns:
        dict, with keys:
            - actions_nopos_in (List[str]): target side input actions without pointer
            - actions_nopos_out (List[str]): target side out actions without pointer
            - actions_pos (List[str]): target side edge pointers associated with output action sequence
            - allowed_cano_actions (List[List[str]]): allowed canonical actions for each action position. This includes
                the last position for "CLOSE" action, whether it is in the "actions" input or not.
            - actions_nodemask (list): a list of 0 or 1 to indicate which actions generate node
            - token_cursors (list, optional): a list of token cursors before each action is applied
    """
    assert actions is not None
    if append_close:
        if actions[-1] != 'CLOSE':
            actions = actions.copy()
            actions.append('CLOSE')
    else:
        if actions[-1] == 'CLOSE':
            actions = actions.copy()
            actions = actions[:-1]

    # This is where to feed in specific machine configs:
    # InterfaceMachine.reset_config(copy_only_string=True, tgt_input_original=False)

    actions_states = InterfaceMachine.get_actions_and_states(actions, src_tokens=tokens)

    return actions_states


def check_actions_file(en_file, actions_file, out_file=None, append_close=False):
    """Run the graph action interface state machine for pairs of English sentences and actions, to check the validity
    of the rules for allowed actions, and output data statistics.

    Args:
        en_file (str): English sentence file path.
        actions_file (str): actions file path.
    Returns:
    """
    avg_num_allowed_actions_pos = 0
    avg_num_allowed_actions_seq = 0
    avg_num_arcs_pos = 0
    avg_num_arcs_not1st_seq = 0
    num_pos = 0
    num_seq = 0
    avg_len_en = 0
    avg_len_actions = 0
    with open(en_file, 'r') as f, open(actions_file, 'r') as g:
        for tokens, actions in tqdm(zip(f, g)):
            if tokens.strip():
                tokens = tokens.strip().split(' ')
                actions = actions.strip().split(' ')
                # assert tokens[-1] == '<ROOT>'
                actions_states = get_actions_states(tokens=tokens, actions=actions, append_close=append_close)
                # breakpoint()
                # get statistics
                allowed_cano_actions = actions_states['allowed_cano_actions']
                num_pos += len(allowed_cano_actions)    # this includes or excludes the last "CLOSE" action
                num_seq += 1
                avg_len_en += len(tokens)
                # avg_len_actions += len(actions)
                avg_len_actions += len(actions_states['actions_nopos_out'])
                avg_num_allowed_actions_pos += sum(map(len, allowed_cano_actions))
                avg_num_allowed_actions_seq += len(list(set.union(*map(set, allowed_cano_actions))))
                actions_cano = list(map(InterfaceMachine.get_canonical_action, actions))
                avg_num_arcs_pos += len(list(filter(lambda act: act.startswith('LA') or act.startswith('RA'),
                                                    actions_cano)))
                avg_num_arcs_not1st_seq += len([1 for a, b in zip(actions_cano, actions_cano[1:])
                                                if (a.startswith('LA') or a.startswith('RA'))
                                                and (b.startswith('LA') or b.startswith('RA'))])

    avg_num_allowed_actions_pos /= num_pos
    avg_num_allowed_actions_seq /= num_seq
    avg_num_arcs_pos /= num_pos
    avg_num_arcs_not1st_seq /= num_seq
    avg_len_en /= num_seq
    avg_len_actions /= num_seq

    print(
        f'number of sequences: {num_seq}, '
        f'number of action tokens ({"in" if append_close else "ex"}cluding CLOSE): {num_pos}',
        file=out_file or sys.stdout)
    print(
        f'average en sentence length (including <ROOT>): {avg_len_en}, '
        f'average actions sequence length (({"in" if append_close else "ex"}cluding CLOSE): {avg_len_actions}',
        file=out_file or sys.stdout)
    print(
        f'average number of arc actions per action token position (excluding CLOSE): {avg_num_arcs_pos}',
        file=out_file or sys.stdout)
    print(
        f'average number of arc actions that are not the 1st arc action inside an arc subsequence per action sequence: '
        f'{avg_num_arcs_not1st_seq}',
        file=out_file or sys.stdout)
    print(
        f'average number of allowed canonical actions per action token position: {avg_num_allowed_actions_pos}',
        file=out_file or sys.stdout)
    print(
        f'average number of allowed canonical actions per action sequence: {avg_num_allowed_actions_seq} (max {len(InterfaceMachine.canonical_actions)})',
        file=out_file or sys.stdout)


if __name__ == '__main__':
    for split in ['train', 'valid', 'test']:

        if split == 'test':
            break

        print('-' * 20)
        print(split + ' data')
        print('-' * 20)

        en_file = f'/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/act-top_src-ct1-npwa_tgt-cp-str/oracle/{split}.utters'
        actions_file = f'/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/act-top_src-ct1-npwa_tgt-cp-str/oracle/{split}.actions.src_copy'
        out_file_path = f'/scratch/work/incremental-interpretation/DATA/processed/smcalflow2.0/act-top_src-ct1-npwa_tgt-cp-str/oracle/{split}.stats'

        out_file = open(out_file_path, 'w')
        check_actions_file(en_file, actions_file, out_file, append_close=False)
        out_file.close()
        os.system(f'cat {out_file_path}')
        print('\n' + f'stats saved to {out_file_path}')

        # break
