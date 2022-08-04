# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Check the action-level probabilities and see if they are well callibrated with the actual distribution.
"""
from dataclasses import dataclass
import json
import math
import os
import pickle
from typing import Dict, List, Tuple

from tqdm import tqdm

from calflow_parsing.io import read_string_sentences
from calflow_parsing.calflow_machine import STRING_MARK, STRING_MARK_END, RA, LA, CalFlowMachine
from calflow_parsing.graph import Graph
from calflow_parsing.calflow_graph import actions_to_graph


rootdir = '../'

prefix_results_path = os.path.join(
    rootdir,
    'SAVE/'
    'exp_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/'
    'models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2/'
    'beam1/'
    'valid-prefixallt.hypos.json'
)
save_suffix = '_p100'

# prefix_results_path = os.path.join(
#     rootdir,
#     'SAVE/'
#     'exp_prefix-last-8ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/'
#     'models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2/'
#     'beam1/'
#     'valid-prefixallt.hypos.json'
# )
# save_suffix = '_last-8ps'

gold_lispress_path = os.path.join(
    rootdir,
    'DATA/'
    'processed/smcalflow2.0/act-top_src-ct1-npwa_tgt-cp-str/oracle/valid.lispress'
)

gold_actions_path = os.path.join(
    rootdir,
    'DATA/'
    'processed/smcalflow2.0/act-top_src-ct1-npwa_tgt-cp-str/oracle/valid.actions'
)

# where to save dumped probabilities
save_dir = './prob_calibration'
os.makedirs(save_dir, exist_ok=True)

# =====

fix_prefix_len_perc = None       # only look at prefix with this percentage length; should be a range e.g. (0.2, 0.5)
fix_prefix_len_abs = None        # only look at prefix with this absolute length
save_suffixx = ''

# fix_prefix_len_perc = (0.9, 1.05)       # only look at prefix with this percentage length; should be a range e.g. (0.2, 0.5)
# fix_prefix_len_abs = None        # only look at prefix with this absolute length
# save_suffixx = '_prefix90up'

fix_prefix_len_perc = (0.0, 0.50001)       # only look at prefix with this percentage length; should be a range e.g. (0.2, 0.5)
fix_prefix_len_abs = None        # only look at prefix with this absolute length
save_suffixx = '_prefix0-50'

# fix_prefix_len_perc = (0.5, 1.05)       # only look at prefix with this percentage length; should be a range e.g. (0.2, 0.5)
# fix_prefix_len_abs = None        # only look at prefix with this absolute length
# save_suffixx = '_prefix50-100'

# fix_prefix_len_perc = (0.0, 0.30001)       # only look at prefix with this percentage length; should be a range e.g. (0.2, 0.5)
# fix_prefix_len_abs = None        # only look at prefix with this absolute length
# save_suffixx = '_prefix0-30'

# fix_prefix_len_perc = (0.3, 0.650001)       # only look at prefix with this percentage length; should be a range e.g. (0.2, 0.5)
# fix_prefix_len_abs = None        # only look at prefix with this absolute length
# save_suffixx = '_prefix30-65'

# fix_prefix_len_perc = (0.65, 1.01)       # only look at prefix with this percentage length; should be a range e.g. (0.2, 0.5)
# fix_prefix_len_abs = None        # only look at prefix with this absolute length
# save_suffixx = '_prefix65-100'

# =====

only_string = False
only_node = False

# different options
# only_string = True
# save_name = f'all_string_probs_n_true{save_suffix}{save_suffixx}.pkl'

# only_node = True                 # NOTE to set this true, need to comment above which has higher priority
# save_name = f'all_node_probs_n_true{save_suffix}{save_suffixx}.pkl'

node_edge = True        # NOTE to set this true, need to comment above which has higher priority
save_name = f'all_node_edge_probs_n_true{save_suffix}{save_suffixx}.pkl'

# ==========


@dataclass(frozen=True)
class ScoredGraph:
    graph: Graph
    node_scores: Dict[int, float]
    arc_scores: Dict[Tuple[int, str, int], float]

    @classmethod
    def get_scored_graph(cls, actions, scores):
        machine = CalFlowMachine()
        machine.apply_actions(actions)
        node_scores = {}
        arc_scores = {}
        for maybe_node, maybe_edge, score in zip(
                machine.action_graph_nodes,
                machine.action_graph_edges,
                scores,
        ):
            if maybe_node is not None:
                node_scores[maybe_node] = score
            if maybe_edge is not None:
                arc_scores[maybe_edge] = score
        return cls(
            graph=machine.graph,
            node_scores=node_scores,
            arc_scores=arc_scores,
        )


if __name__ == '__main__':
    prefix_results = json.load(open(prefix_results_path, 'rb'))
    gold_lispress_list = read_string_sentences(gold_lispress_path)
    gold_actions_list = read_string_sentences(gold_actions_path)

    num_total = len(gold_actions_list)

    # breakpoint()

    probs_and_true = []

    for i in tqdm(range(num_total)):
        predictions = prefix_results[str(i)]['predictions']
        gold_lispress = gold_lispress_list[i]
        gold_actions = gold_actions_list[i].split()    # List

        num_prefixes = len(predictions)
        for p in range(num_prefixes):
            if fix_prefix_len_perc is not None:
                if not ((p / (num_prefixes - 1)) >= fix_prefix_len_perc[0]
                        and (p / (num_prefixes - 1)) < fix_prefix_len_perc[1]):
                    continue

            if fix_prefix_len_abs is not None:
                if not p == fix_prefix_len_abs:
                    continue

            predicted_actions = predictions[str(p)][0]['actions']        # [0] is for beam number. already a List
            predicted_scores = predictions[str(p)][0]['positional_scores']    # [0] is for beam number. already a List
            # NOTE scores are log probs, with one more step at the end

            if only_string:
                # only look at string nodes
                string_within = False
                for act, ps in zip(predicted_actions, predicted_scores):
                    if act == STRING_MARK and not string_within:
                        string_within = True
                        continue

                    if string_within and act == STRING_MARK_END:
                        string_within = False
                        continue

                    if string_within:
                        # inside the string
                        # NOTE do not count the future mask "<mask>" token, which should be of high probability but
                        #      never appear in the gold graph
                        if act == '<mask>':
                            ...
                        else:
                            if act in gold_actions:
                                probs_and_true.append((math.exp(ps), 1))
                            else:
                                probs_and_true.append((math.exp(ps), 0))

            elif only_node:
                for act, ps in zip(predicted_actions, predicted_scores):
                    if act.startswith((RA, LA)):
                        continue
                    else:
                        if act == '<mask>':
                            continue
                        else:
                            if act in gold_actions:
                                probs_and_true.append((math.exp(ps), 1))
                            else:
                                probs_and_true.append((math.exp(ps), 0))

            elif node_edge:
                scored_graph = ScoredGraph.get_scored_graph(predicted_actions, predicted_scores)
                gold_graph = actions_to_graph(gold_actions)
                gold_graph_nodes = set(gold_graph.nodes.values())
                gold_graph_edges = set((gold_graph.nodes[s], r, gold_graph.nodes[t]) for s, r, t in gold_graph.edges)
                for nid, ps in scored_graph.node_scores.items():
                    if nid not in scored_graph.graph.nodes:
                        continue
                    if ('<mask>' not in scored_graph.graph.nodes[nid] and
                            scored_graph.graph.nodes[nid] in gold_graph_nodes):
                        probs_and_true.append((math.exp(ps), 1))
                    else:
                        probs_and_true.append((math.exp(ps), 0))
                for (s, r, t), ps in scored_graph.arc_scores.items():
                    if ('<mask>' not in scored_graph.graph.nodes[s] and '<mask>' not in scored_graph.graph.nodes[t] and
                            (scored_graph.graph.nodes[s], r, scored_graph.graph.nodes[t]) in gold_graph_edges):
                        probs_and_true.append((math.exp(ps), 1))
                else:
                    probs_and_true.append((math.exp(ps), 0))

            else:
                ...

    probs_and_true = sorted(probs_and_true)    # default is using the first element to sort in ascending order
    print(f'sorted probabilities with their actual binary labels obtained. Total num: {len(probs_and_true)}')

    # save results
    save_path = os.path.join(save_dir, save_name)
    pickle.dump(probs_and_true, open(save_path, 'wb'))
    print(f'Results saved at {save_path}')

    # breakpoint()
