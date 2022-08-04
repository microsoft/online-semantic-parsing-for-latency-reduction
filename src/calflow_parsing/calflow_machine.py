# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""From the graph, create a sequence of node-edge actions for building the graph (following certain order).
Also the reverse process, i.e. build the graph based on a sequence of actions.
Both processes could be run stepwise.
"""
import copy
import re
from typing import List, Dict

from calflow_parsing.graph import Graph


# arc/edge symbols
RA = '-RA-'
LA = '-LA-'
# STRING_MARK = '"'
STRING_MARK = '<str>'    # could also have a pair <str> and </str>
STRING_MARK_END = '</str>'

# matching regex
la_reg = re.compile(r'-LA-\((.*),(.*)\)')
ra_reg = re.compile(r'-RA-\((.*),(.*)\)')
arc_reg = re.compile(r'-[RL]A-\((.*),(.*)\)')
la_nopointer_reg = re.compile(r'-LA-\((.*)\)')
ra_nopointer_reg = re.compile(r'-RA-\((.*)\)')
arc_nopointer_reg = re.compile(r'-[RL]A-\((.*)\)')


class CalFlowOracle:
    """Create oracle action sequence to build the graph."""
    NODE_ORDER = 'top-down'        # ['top-down', 'bottom-up']
    EDGE_ORDER = 'close-to-far'    # ['close-to-far', 'far-to-close']
    # EDGE_ORDER = 'far-to-close'

    def __init__(self, gold_graph: Graph = None):
        self.gold_graph = gold_graph

    @classmethod
    def order_nodes(cls, graph: Graph = None, order: str = None) -> List[int]:
        """Return an order the nodes for graph generation."""
        order = order or cls.NODE_ORDER    # default to class order if not specified

        if order == 'top-down':
            # NOTE the default graph built from the calflow program is in bottom-up order
            nid_sorted = sorted([nid for nid in graph.node_ids], reverse=True)
        elif order == 'bottom-up':
            # NOTE the default graph built from the calflow program is in bottom-up order
            nid_sorted = sorted([nid for nid in graph.node_ids])
        else:
            raise NotImplementedError    # TODO add other order modules

        return nid_sorted

    @classmethod
    def graph_to_actions(cls, graph: Graph = None, order: str = None) -> List[str]:
        """Build the node-edge action sequence for constructing the graph."""
        # graph = graph or self.gold_graph
        # we require explicitly input the graph
        assert graph

        # order the nodes
        nid_sorted = cls.order_nodes(graph, order)

        actions = []
        action_idx_to_nids = []
        action_idx_cur = 0
        for nid in nid_sorted:
            # add node
            if graph.nodes[nid].startswith('"'):
                # string values: with quotes " " and span a few steps
                assert graph.nodes[nid].endswith('"')
                string_tokenized = [STRING_MARK] + graph.nodes[nid][1:-1].strip().split() + [STRING_MARK_END]
                for i, s in enumerate(string_tokenized):
                    actions.append(s)
                    action_idx_cur += 1
                    if i == 0:
                        action_idx_to_nids.append(nid)
                    else:
                        action_idx_to_nids.append(None)
            else:
                # other nodes: all take a single step
                actions.append(graph.nodes[nid])
                action_idx_cur += 1
                action_idx_to_nids.append(nid)

            # add edges
            num_previous_actions = len(action_idx_to_nids)
            if cls.EDGE_ORDER == 'far-to-close':
                previous_action_ids = range(num_previous_actions)
            elif cls.EDGE_ORDER == 'close-to-far':
                previous_action_ids = list(range(num_previous_actions))[::-1]
            else:
                raise NotImplementedError

            for act_idx in previous_action_ids:
                node_idx = action_idx_to_nids[act_idx]
                for (s, r, t) in graph.edges:
                    if s == node_idx and t == nid:
                        # right arc
                        actions.append(f'{RA}({act_idx},{r})')
                        action_idx_to_nids.append(None)
                        action_idx_cur += 1
                    elif s == nid and t == node_idx:
                        # left arc
                        actions.append(f'{LA}({act_idx},{r})')
                        action_idx_to_nids.append(None)
                        action_idx_cur += 1
                    else:
                        pass

        return actions

    def get_fixed_action(self):
        """Return a single action based on a fixed oracle."""
        ...

    def get_actions_and_scores(self):
        """Return a set of valid actions and their scores."""
        ...
        # TODO finish this


class CalFlowMachine:
    """Recover the graph from the action sequence."""

    # canonical actions without the detailed node/edge labels and action properties (e.g. arc pointer value)
    canonical_actions = ['NODE',
                         'NODE_SEP',
                         'NODE_SEP_END',
                         'LA',
                         'RA',
                         'CLOSE']    # NOTE 'CLOSE' action is not explicitly written in the data

    @classmethod
    def canonical_action_to_dict(cls, vocab):
        """Map the canonical actions to ids in a vocabulary, each canonical action corresponds to a set of ids.

        CLOSE is mapped to eos </s> token.
        """
        canonical_act_ids = dict()
        vocab_act_count = 0
        for i in range(len(vocab)):
            # NOTE can not directly use "for act in vocab" -> this will never stop since no stopping iter implemented
            act = vocab[i]
            cano_act = cls.get_canonical_action(act) if i != vocab.eos() else 'CLOSE'
            if cano_act in cls.canonical_actions:
                vocab_act_count += 1
                canonical_act_ids.setdefault(cano_act, []).append(i)
        # print for debugging
        # print(f'{vocab_act_count} / {len(vocab)} tokens in action vocabulary mapped to canonical actions.')
        return canonical_act_ids

    def __init__(self):
        # graph
        self.graph = Graph()

        # intermediate states
        self.string_node_alignment: Dict[int, List[int]] = {}
        self.string_within = False
        self.string_current = []
        self.current_node_id = 0    # 0 represents None (0 is reserved too)
        self.step = 0    # number of steps (applied actions) so far
        self.is_closed = False    # the end of the graph
        # memories for possible external use
        self.string_node_mask = []
        self.action_to_node_id = []
        self.action_history = []

        # for alignment between the actions and the constructed graph
        self.action_graph_nodes = []    # for each action, which node does it correspond to (otherwise None)
        self.action_graph_edges = []    # for each action, which edge does it correspond to (otherwise None)

    def __deepcopy__(self, memo):
        """
        Manual deep copy of the machine.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    @classmethod
    def get_canonical_action(cls, action):
        """Get the canonical action form, by stripping out labels, pointers, etc."""
        # NOTE do not use this, as 'LA' would be returned directly as an edge, but it's a node name
        # if action in cls.canonical_actions:
        #     return action
        # NOTE do explicit matching
        if action in ['CLOSE', '<CLOSE>']:
            return 'CLOSE'

        if action == STRING_MARK:
            # node separation
            return 'NODE_SEP'
        if action == STRING_MARK_END:
            # node separation end
            return 'NODE_SEP_END'
        # NOTE need to deal with both '-LA-(pos,label)' and '-LA-(label)',
        #      as in the vocabulary the pointers are peeled off
        if la_reg.match(action) or la_nopointer_reg.match(action):
            return 'LA'
        if ra_reg.match(action) or ra_nopointer_reg.match(action):
            return 'RA'
        # if arc_reg.match(action) or arc_nopointer_reg.match(action):
        #     return 'EDGE'
        return 'NODE'

    def get_valid_canonical_actions(self):
        """Get the valid canonical actions for the next step."""
        valid_cano_actions = []
        # multi-step node: need to close
        if self.string_within:
            valid_cano_actions.extend(['NODE'])
            if self.string_current:
                # only close string if it is not empty
                valid_cano_actions.extend(['NODE_SEP_END'])
            return valid_cano_actions

        # single step node, multi-step node mark: any time
        valid_cano_actions.extend(['NODE', 'NODE_SEP'])

        # CLOSE: any time outside of the string
        valid_cano_actions.append('CLOSE')

        # edges: any time after there are two nodes (assuming no self-loop)
        if sum(self.get_actions_nodemask()) >= 2:
            valid_cano_actions.extend(['LA', 'RA'])
        return valid_cano_actions

    def apply_action(self, action: str):
        assert not self.is_closed, 'graph construction is closed; cannot apply any actions.'

        if re.match(r'CLOSE', action):
            self.is_closed = True

        if not self.string_within and action == STRING_MARK:
            # start of a string node
            self.string_within = True
            self.string_node_mask.append(1)
            # update node states: use the starting position as reference
            self.current_node_id += 1
            self.action_to_node_id.append(self.current_node_id)

            # action-graph alignment
            self.action_graph_nodes.append(self.current_node_id)

        elif self.string_within and action == STRING_MARK_END:
            # end of a string node
            self.string_within = False
            self.string_node_mask.append(1)
            node_name = '"' + ' '.join(self.string_current) + '"'

            # add the node
            self.graph.add_node(node_id=self.current_node_id, node_name=node_name)
            # NOTE only the starting position of the string is used for the node reference
            # update string node to action alignment
            self.string_node_alignment[self.current_node_id] = list(range(self.step - len(self.string_current) - 1,
                                                                          self.step + 1))
            # reset the string memory
            self.string_current = []

            # action-graph alignment
            self.action_graph_nodes.append(self.current_node_id)

        else:

            if self.string_within:
                # inside of a string node
                self.string_node_mask.append(1)
                self.string_current.append(action)

                # action-graph alignment
                self.action_graph_nodes.append(self.current_node_id)

            elif ra_reg.match(action):
                # right arc -->
                idx, label = ra_reg.match(action).groups()
                idx = int(idx)
                # add edge
                self.graph.add_edge((self.action_to_node_id[idx], label, self.current_node_id))

                # action-graph alignment
                self.action_graph_edges.append((self.action_to_node_id[idx], label, self.current_node_id))

            elif la_reg.match(action):
                # left arc <--
                idx, label = la_reg.match(action).groups()
                idx = int(idx)
                # add edge
                self.graph.add_edge((self.current_node_id, label, self.action_to_node_id[idx]))

                # action-graph alignment
                self.action_graph_edges.append((self.current_node_id, label, self.action_to_node_id[idx]))

            else:
                # single step node names
                node_name = action
                # add node
                self.current_node_id += 1
                self.graph.add_node(node_id=self.current_node_id, node_name=node_name)
                self.action_to_node_id.append(self.current_node_id)

                # action-graph alignment
                self.action_graph_nodes.append(self.current_node_id)

        # update states
        self.step += 1
        self.action_history.append(action)
        if len(self.string_node_mask) < len(self.action_history):
            self.string_node_mask.append(0)
            assert len(self.string_node_mask) == len(self.action_history)
        if len(self.action_to_node_id) < len(self.action_history):
            self.action_to_node_id.append(None)
            assert len(self.action_to_node_id) == len(self.action_history)

        if len(self.action_graph_nodes) < len(self.action_history):
            self.action_graph_nodes.append(None)
            assert len(self.action_graph_nodes) == len(self.action_history)
        if len(self.action_graph_edges) < len(self.action_history):
            self.action_graph_edges.append(None)
            assert len(self.action_graph_edges) == len(self.action_history)

        return

    def apply_actions(self, actions: List[str]):
        for action in actions:
            self.apply_action(action)

        return

    def get_actions_nodemask(self):
        actions_nodemask = [1 if x is not None else 0 for x in self.action_to_node_id]
        return actions_nodemask


def peel_pointer(action, pad=-1):
    """Peel off the pointer value from arc actions"""
    if arc_reg.match(action):
        # LA(pos,label) or RA(pos,label)
        action, properties = action.split('(')
        properties = properties[:-1]    # remove the ')' at last position
        properties = properties.split(',')    # split to pointer value and label
        pos = int(properties[0].strip())
        label = properties[1].strip()    # remove any leading and trailing white spaces
        action_label = action + '(' + label + ')'
        return (action_label, pos)
    else:
        return (action, pad)


def join_action_pointer(action, pos):
    """Join action label and pointer value.

    Args:
        action (str): action label without pointer
        pos (int or str): pointer value

    Return:
        action_complete (str): complete action label
    """
    if action.startswith(LA) or action.startswith(RA):
        action_parts = action.split('(')
        assert len(action_parts) == 2
        assert int(pos) >= 0
        action_complete = f'{action_parts[0]}({pos},{action_parts[1]}'
    else:
        action_complete = action
    return action_complete
