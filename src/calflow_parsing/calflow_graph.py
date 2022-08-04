# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
From the raw SMCalFlow data, convert the programs into graphs, based on the lispress order (treated as
the oracle order), and create the node-edge action sequences for target generation.
"""
import argparse
import copy
from dataclasses import replace
from json import JSONDecodeError, loads
import os
import sys
import re
from typing import DefaultDict, Dict, Tuple, Iterator, List, TextIO

import jsons
from tqdm import tqdm

from dataflow.core.dialogue import Dialogue, Turn, TurnId
from dataflow.core.program import Program, TypeName
from dataflow.core.program_utils import (
    get_named_args,
    is_struct_op_schema,
    mk_value_op,
    mk_struct_op,
    mk_call_op
    )
from dataflow.core.lispress import (
    Lispress,
    parse_lispress,
    lispress_to_program,
    program_to_lispress,
    render_compact,
    op_to_lispress,
    type_args_to_lispress,
    _key_to_named_arg,
    _named_arg_to_key,
    # _roots_and_reentrancies
    )
from dataflow.core.program import roots_and_reentrancies as _roots_and_reentrancies    # adapt changes in later update of the repo
from dataflow.core.sexp import sexp_to_str
from dataflow.onmt_helpers.create_onmt_text_data import create_context_turns, create_source_str
from calflow_parsing.io import write_tokenized_sentences, write_string_sentences
from calflow_parsing.graph import Graph, Infeasible
from calflow_parsing.calflow_machine import CalFlowOracle, CalFlowMachine
from calflow_parsing.tokenizer import tokenize, detokenize, detokenize_treedst


_long_number_regex = re.compile("^([0-9]+)L$")

# We assume all dialogues start from turn 0.
# This is true for MultiWoZ and CalFlow datasets.
_MIN_TURN_INDEX = 0

_CONVERSION_ERROR = '(ConversionErrorGraphToLispress)'

count = 0    # for temporary use


class ProgramGraph:
    """Conversion between program and graph. We also have lispress <-> program, and so lispress <-> graph.
    """
    def __init__(self, type_args_coversion='atomic', dataset='smcalflow'):
        # how to handle type_args in the graph
        self.type_args_conversion = type_args_coversion    # choose between ['atomic', 'singlenode', 'mergednode']
        # TODO implement the other methods besides 'atomic'

        # some variations for different datasets
        assert dataset in ['smcalflow', 'treedst']
        self.dataset = dataset

    @staticmethod
    def type_args_subgraph(type_args: List[TypeName],
                           initial_graph=Graph(nodes={0: 'ROOT'}, root=0),
                           initial_node_id=0,
                           ) -> Graph:
        """Convert the type_args (marked by ^ in SMCalFlow 2.0) to a subgraph. The type_args could be more than one,
        and could be recursive.
        Each type_arg is a new node, with an edge from the root (the function) or the type_arg in previous level.

        Args:
            type_args: type arguments
            initial_graph: graph to extend upon
            initial_node_id: node_id to start growing the subgraph

        Returns:
            Graph: new or modified graph with the type_args nodes
        """
        if not type_args:
            # None or []
            return initial_graph

        # # sanity check: if there are more than one type_arg at any nested level -> Nope for both
        # #               SMCalFlow 2.0 train and valid
        # if len(type_args) > 1:
        #     print(type_args)
        #     breakpoint()

        parent_id_cur = initial_node_id
        node_id_cur = max(initial_graph.node_ids) + 1    # to ensure non-overlapping new node ids
        for idx, targ in enumerate(type_args):
            # base type_args
            initial_graph.add_node(node_id_cur, targ.base)
            initial_graph.add_edge((parent_id_cur, f':type-arg{idx}', node_id_cur))
            parent_id_top = parent_id_cur    # save the parent pointer at the top level
            parent_id_cur = node_id_cur    # move the parent pointer to the new node immediately for depth growing
            node_id_cur += 1

            # recursive type_args (could be empty); `initial_graph` is modified in place
            _ = ProgramGraph.type_args_subgraph(targ.type_args,
                                                initial_graph=initial_graph,
                                                initial_node_id=parent_id_cur)

            parent_id_cur = parent_id_top

        return initial_graph

    @staticmethod
    def type_args_single_node(type_args: List[TypeName],
                              initial_graph=Graph(nodes={0: 'ROOT'}, root=0),
                              initial_node_id=0,
                              ) -> Graph:
        """Convert the type_args (marked by ^ in SMCalFlow 2.0) to a single node (with a single edge from the
        function node it modifies).

        Args:
            type_args (List[TypeName]): [description]
            initial_graph ([type], optional): [description]. Defaults to Graph(nodes={0: 'ROOT'}, root=0).
            initial_node_id (int, optional): [description]. Defaults to 0.

        Returns:
            Graph: [description]
        """
        ...

    @staticmethod
    def type_args_subgraph_expression(graph: Graph) -> Tuple[Dict[int, List[TypeName]], set]:
        """Extract type_args from a graph. Reverse process of `type_args_subgraph`.

        Args:
            graph (Graph): [description]

        Returns:
            type_args (Dict[int, List[TypeName]]): dictionary of node ids and their type args
            type_noes (set): set of type_args node ids
        """
        children_type_args = {}
        for s, r, t in graph.edges:
            if r.startswith(':type'):
                children_type_args.setdefault(s, []).append((r, t))
        # sort the type_args
        for s, v in children_type_args.items():
            # numerical order (although in the data each node has at most one type_args)
            children_type_args[s] = sorted(v, key=lambda x: x[0])

        # create nested type_args
        type_args = {}
        type_nodes = set()
        while children_type_args:
            # wrapping with list() for looping over chaning dictionary
            for s, v in list(children_type_args.items()):
                # no nested, or all nested have been converted
                if all(t not in children_type_args for r, t in v) \
                  or all(t in type_args for r, t in v):
                    type_names = []
                    for r, t in v:
                        if t not in type_args:
                            type_names.append(TypeName(base=graph.nodes[t], type_args=[]))
                        else:
                            type_names.append(TypeName(base=graph.nodes[t], type_args=copy.deepcopy(type_args[t])))
                            # NOTE assume one node has at most one type_arg parent, so we can
                            # safely delete after being used once
                            del type_args[t]
                        # update the set of typed nodes
                        type_nodes.add(t)
                    # update the type_args dictionary
                    type_args[s] = type_names
                    # delete the processed node
                    del children_type_args[s]

        return type_args, type_nodes

    def program_to_graph(self, program: Program) -> Graph:
        """Build a directed graph from a calflow program.

        Args:
            program (Program): [description]

        Returns:
            Graph: [description]
        """
        if len(program.expressions) == 0:
            return Graph()

        node_id_to_exp_id = {}
        exp_id_to_node_id = {}
        nodes = {}
        edges = []

        node_id_cur = 1    # 0 is reserved for root node
        node_id_base = None    # for the base function for type args
        for expression in program.expressions:
            # 1-to-1 mapping (NOTE type args are not included here)
            node_id_to_exp_id[node_id_cur] = expression.id
            exp_id_to_node_id[expression.id] = node_id_cur

            node_name = sexp_to_str(op_to_lispress(expression.op))

            # sanity check: `node_name` only contains 1 token, except for strings with " "
            if node_name.startswith('"'):
                assert node_name.endswith('"')
                # " " are untokenized
                node_tokens = node_name[1:-1].split()
                string_node = True
            else:
                node_tokens = node_name.split()
                string_node = False
            if len(node_tokens) > 1:
                assert string_node, 'only string nodes should have more than 1 token'

            # add nodes
            nodes[node_id_cur] = node_name
            # record the current function node
            node_id_base = node_id_cur

            # add edges
            named_args = get_named_args(expression)
            # NOTE this also numbers the positional arguments as "arg{i}" for CallLikeOp.
            #      for BuildStructOp, the argument names could be `None`

            # if all args are named (i.e., not positional), sort them alphabetically
            # TODO in principle, we could get mixed positional and names arguments,
            #   but for now that doesn't happen in SMCalFlow 2.0 so this code is good
            #   enough. This code also only works for functions with named arguments
            #   that have upper case names, which again happens to work for SMCalFlow
            #   2.0.
            # (from function `dataflow.core.lispress._program_to_unsugared_lispress`)

            is_positional = [k is None for k, _ in named_args]
            has_positional = any(is_positional)

            # # sanity check: whether all positional or all named -> The answer is Nope
            # if has_positional:
            #     # assert all(is_positional), 'there are mixture of positional and named arguments'
            #     if not all(is_positional):
            #         print(named_args)
            #         breakpoint()

            # NOTE there are mixed positional and named arguments
            if has_positional:
                # sanity check: all the positional args are before named args, so that there is no confusion
                #               when we label arcs with ":arg{i}" for positional args, the numbering is consecutive
                num_positional = sum(is_positional)

                assert not any([(b - a) == 1 for a, b in zip(is_positional, is_positional[1:])]), \
                    'positional arguments come after named arguments'   # should only be 0 or -1, but not 1

            for idx, (arg_name, arg_id) in enumerate(named_args):
                if arg_name is not None:
                    edge_label = _key_to_named_arg(arg_name)    # add ":" in front
                else:
                    edge_label = _key_to_named_arg(f'arg{idx}')    # add ":" in front
                edges.append((node_id_base, edge_label, exp_id_to_node_id[arg_id]))

            # add type args: nodes and type edges
            if expression.type_args is not None:
                # use the helper function
                # op_type_args_lispress = type_args_to_lispress(expression.type_args)    # list
                # use customized processing for more explicit control
                assert isinstance(expression.type_args, list)

                # # debug: check cases with more than 1 type_args -> no such cases on SMCalFlow 2.0 here
                # if len(expression.type_args) > 1:
                #     from dataflow.core.lispress import program_to_lispress, render_pretty
                #     print(render_pretty(program_to_lispress(program)))
                #     breakpoint()

                # # debug: check the nested type_args
                # for idx, targ in enumerate(expression.type_args):
                #     if len(targ.type_args) > 0:
                #         print(targ)
                #         breakpoint()

                if self.type_args_conversion == 'atomic':
                    subgraph = ProgramGraph.type_args_subgraph(expression.type_args,
                                                               initial_graph=Graph(
                                                                   nodes={node_id_base: node_name},
                                                                   root=node_id_base
                                                                   ),
                                                               initial_node_id=node_id_base)
                    # remove the parent node in the new node dict
                    new_nodes = copy.deepcopy(subgraph.nodes)
                    new_nodes.pop(node_id_base)
                    # add the new nodes and edges
                    nodes.update(new_nodes)
                    edges += subgraph.edges
                    node_id_cur = max(nodes.keys())
                else:
                    raise NotImplementedError

            # increase the node id for the next node
            node_id_cur += 1

        # for root node
        roots, reentrancies = _roots_and_reentrancies(program)    # both return a set
        assert roots, "program must have at least one root"

        # # sanity check: multiple roots
        # if len(roots) > 1:
        #     from dataflow.core.lispress import program_to_lispress, render_pretty
        #     print('\n')
        #     print(render_pretty(program_to_lispress(program)))
        #     print('Multiple roots:')
        #     print(roots)
        #     print([nodes[exp_id_to_node_id[i]] for i in roots])
        #     print('\n')
        #     breakpoint()

        roots = sorted(list(roots), reverse=True)
        root = exp_id_to_node_id[roots[0]]    # take the one with largest id as root

        graph = Graph(nodes, edges, root)

        return graph

    def graph_to_program(self, graph: Graph) -> Program:
        """Convert a directed graph into a Program representation with a list of "expressions", serving as the
        intermediate step towards converting back to the standard lispress format ("standard" as used by SMCalFlow
        dataset, with special orders and printing of re-entrancy and multiple roots.)

        Args:
            graph (Graph): [description]

        Returns:
            Program: [description]
        """
        # order the nodes bottom-up to build expressions
        ordered_node_ids = graph.ordering_bottom_up()
        # get all the args (except type_args)
        children_pos_args = {}
        children_named_args = {}
        for s, r, t in graph.edges:
            if not r.startswith(':type'):
                if r.startswith(':arg'):
                    children_pos_args.setdefault(s, []).append((r, t))
                else:
                    children_named_args.setdefault(s, []).append((r, t))

        # sort the args
        for s, v in children_pos_args.items():
            # numerical order
            children_pos_args[s] = sorted(v)
        for s, v in children_named_args.items():
            # alphabetical order
            children_named_args[s] = sorted(v)

        # extract the type_args
        if self.type_args_conversion == 'atomic':
            type_args, type_nodes = ProgramGraph.type_args_subgraph_expression(graph)
        else:
            raise NotImplementedError

        # states when building the program/expressions
        idx = 0
        node_id_to_idx: Dict[int, int] = {}    # node_id to expression_id map
        expressions = []

        for nid in ordered_node_ids:
            if nid in type_nodes:
                # do not add a new expression for type nodes
                continue

            node_name = graph.nodes[nid]

            # check for ValueOp
            m = _long_number_regex.match(node_name)
            if m is not None:
                n = m.group(1)
                expr, idx = mk_value_op(value=int(n), schema="Long", idx=idx)
                expressions.append(expr)
                node_id_to_idx[nid] = idx    # the newly built expression idx is returned by the `mk_valu_op` function
            else:
                # bare value
                try:
                    value = loads(node_name)
                    known_value_types = {
                        str: "String",
                        float: "Number",
                        int: "Number",
                        bool: "Boolean",
                    }
                    schema = known_value_types[type(value)]
                    expr, idx = mk_value_op(value=value, schema=schema, idx=idx)
                    expressions.append(expr)
                    node_id_to_idx[nid] = idx
                except (JSONDecodeError, KeyError):
                    # not a ValueOp

                    # check for BuildStructOp
                    if is_struct_op_schema(node_name):
                        # arg name and value index pairs
                        kvs = []
                        # NOTE positional arg names are set to `None`
                        if pos_args := children_pos_args.get(nid):
                            # NOTE `node_id_to_idx[t]` exists, guaranteed by the bottom-up order
                            kvs += [(None, node_id_to_idx[t]) for r, t in pos_args]
                        if named_args := children_named_args.get(nid):
                            kvs += [(_named_arg_to_key(r), node_id_to_idx[t]) for r, t in named_args]
                        expr, idx = mk_struct_op(node_name, kvs, idx)

                    else:
                        # # CallLikeOp
                        # # assumption: all positional args
                        # args: List[int] = []
                        # if pos_args := children_pos_args.get(nid):
                        #     # NOTE `node_id_to_idx[t]` exists, guaranteed by the bottom-up order
                        #     args += [node_id_to_idx[t] for r, t in pos_args]
                        # # TODO here the "assert" below hit errors with some bottom-up generation model
                        # # debug
                        # # if children_named_args.get(nid):
                        # #     print('\n')
                        # #     print(children_named_args.get(nid))
                        # #     print('\n')
                        # #     breakpoint()
                        # assert not children_named_args.get(nid), f'CallLikeOp {node_name} should not have named_args'
                        # expr, idx = mk_call_op(node_name, args, idx=idx)

                        """SMCalFlow assumptions:
                        BuildStructOp schemas begin with a capital letter, with both positional and named args;
                        CallLikeOp schema with all positional args;
                        NOTE The above does not hold true for some bottom-up model generation, and for TreeDST data,
                             this is not true even in the gold data.
                        """
                        if not children_named_args.get(nid):
                            # --> pure CallLike Op
                            args: List[int] = []
                            if pos_args := children_pos_args.get(nid):
                                # NOTE `node_id_to_idx[t]` exists, guaranteed by the bottom-up order
                                args += [node_id_to_idx[t] for r, t in pos_args]
                            expr, idx = mk_call_op(node_name, args, idx=idx)
                        else:
                            # --> more like BuildStructOp
                            # arg name and value index pairs
                            kvs = []
                            # NOTE positional arg names are set to `None`
                            if pos_args := children_pos_args.get(nid):
                                # NOTE `node_id_to_idx[t]` exists, guaranteed by the bottom-up order
                                kvs += [(None, node_id_to_idx[t]) for r, t in pos_args]
                            named_args = children_named_args.get(nid)
                            kvs += [(_named_arg_to_key(r), node_id_to_idx[t]) for r, t in named_args]
                            expr, idx = mk_struct_op(node_name, kvs, idx)

                    # type_args
                    if type_names := type_args.get(nid):
                        expr = replace(expr, type_args=type_names)

                    expressions.append(expr)
                    node_id_to_idx[nid] = idx

        program = Program(expressions)
        return program

    def lispress_to_graph(self, lispress: Lispress) -> Graph:
        program, _ = lispress_to_program(parse_lispress(lispress), idx=0)
        graph = self.program_to_graph(program)
        return graph

    def graph_to_lispress(self, graph: Graph) -> Lispress:
        try:
            program = self.graph_to_program(graph)
        except Infeasible:
            return _CONVERSION_ERROR
        lispress = program_to_lispress(program)
        # TODO check render strings and normalization
        lispress = render_compact(lispress)
        return lispress

    def graph_tokenize_string(self, graph: Graph) -> Graph:
        """Tokenize the strings (they are untokenized including the quotes, e.g. "doctor's meeting") in the graph."""
        graph = copy.deepcopy(graph)

        for nid, node_name in graph.nodes.items():
            if node_name.startswith('"'):
                if self.dataset == 'smcalflow':
                    assert node_name.endswith('"') and len(node_name) >= 3
                elif self.dataset == 'treedst':
                    # NOTE disable the above `assert`for non-empty string name. In TreeDST there are lispress with "" empty
                    # string, e.g.
                    # '(plan (revise (^(Unit) Path.apply "Create") (^(Unit) Path.apply "") (lambda (^Unit x0) x0)))'
                    assert node_name.endswith('"')
                else:
                    raise NotImplementedError
                graph.nodes[nid] = '"' + ' '.join(tokenize(node_name[1:-1])) + '"'

        return graph

    def graph_detokenize_string(self, graph: Graph) -> Graph:
        """Detokenize the strings (as node names, e.g. "doctor 's meeting") in the graph."""
        graph = copy.deepcopy(graph)

        for nid, node_name in graph.nodes.items():
            if node_name.startswith('"'):
                if not (node_name.endswith('"') and len(node_name) >= 3):
                    import warnings
                    warnings.warn('The string node value (node_name should be "xxx") does not look legit (e.g. empty).')

                if self.dataset == 'smcalflow':
                    graph.nodes[nid] = '"' + detokenize(node_name[1:-1].split()) + '"'
                elif self.dataset == 'treedst':
                    graph.nodes[nid] = '"' + detokenize_treedst(node_name[1:-1].split()) + '"'
                else:
                    raise NotImplementedError

        return graph


def actions_to_graph(actions: List[str]) -> Graph:
    """Build the graph by executing the actions."""
    machine = CalFlowMachine()
    machine.apply_actions(actions)
    graph = machine.graph
    return graph


# graph_to_lispress = ProgramGraph(type_args_coversion='atomic').graph_to_lispress

pg_methods = ProgramGraph(type_args_coversion='atomic')


def graph_to_lispress(graph: Graph):
    from calflow_parsing.calflow_graph_lispress import _CONVERSION_ERROR
    try:
        lispress = pg_methods.graph_to_lispress(graph)
    except RuntimeError as re:
        lispress = _CONVERSION_ERROR
        print(re.args[0], file=sys.stderr)

    return lispress


lispress_to_graph = ProgramGraph(type_args_coversion='atomic').lispress_to_graph


def create_graph_actions_from_dialogue_turn(turn: Turn, program_graph_methods: ProgramGraph, graph_order: str = None):
    """Create graph actions from a dialogue turn, following certain node orders."""
    # turn.lispress -> string, untokenized lispress
    # turn.tokenized_lispress() -> list of string, tokenized lispress
    # ' '.join(turn.tokenized_lispress()) -> string, tokenized lispress
    # from dataflow.core.lispress import parse_lispress; parse_lispress(turn.lispress) -> nested list, parsed lispress
    # turn.program() -> Program (from dataflow.core.program import Program)

    # # debuging the pipeline with simply linearized program
    # return turn.tokenized_lispress()

    # try:
    #     treedst_test = turn.program()
    # except:
    #     breakpoint()
    graph = program_graph_methods.program_to_graph(turn.program())
    # OR
    # graph = program_graph_methods.lispress_to_graph(turn.lispress)

    # # debug: print at each turn
    # print(graph)
    # print(turn.tokenized_lispress())
    # breakpoint()

    # # debug: round trip program -> graph -> program; all programs match
    # program = program_graph_methods.graph_to_program(graph)
    # if turn.program() != program:
    #     breakpoint()

    # return [v for k, v in graph.nodes.items()]

    # try:
    graph_tokstr = program_graph_methods.graph_tokenize_string(graph)
    # except:
    #     breakpoint()
    # graph_tokstr = graph

    # # === there are corner cases when the detokenization can not fully recover the original string
    # if not (graph_back := program_graph_methods.graph_detokenize_string(graph_tokstr)).is_identical(graph):
    #     print('Graph tokenization -> detokenization not matching original')
    #     print(graph_back)
    #     print(graph)
    #     breakpoint()

    actions = CalFlowOracle.graph_to_actions(graph_tokstr, order=graph_order)

    # debug: reverse process: actions -> graph -> program -> lispress
    graph_tokstr_rec = actions_to_graph(actions)

    # program_rec = program_graph_methods.graph_to_program(graph_rec)
    # if turn.program() != program_rec:
    #     # NOTE this now becomes different, as the recovered graph has different node orders
    #     breakpoint()
    if not graph_tokstr.is_identical(graph_tokstr_rec):
        print('Recovered graph not matching')
        breakpoint()

    graph_rec = program_graph_methods.graph_detokenize_string(graph_tokstr_rec)
    # graph_rec = graph_tokstr_rec

    # try:
    lispress_rec = program_graph_methods.graph_to_lispress(graph_rec)
    # except:
    #     print(graph_rec)
    #     breakpoint()
    global count
    if turn.lispress != lispress_rec:
        # print('\n')
        # print('origina lispress:')
        # print(turn.lispress)
        # print('recovered lispress:')
        # print(lispress_rec)
        # print('\n')
        # breakpoint()

        count += 1

    # TODO tokenized lispress

    # return actions, graph
    return actions, graph_tokstr


def create_source_contents_for_turn(
    turn_lookup: Dict[int, Turn],
    curr_turn_index: int,
    curr_turn: Turn,
    *,
    num_context_turns: int,
    min_turn_index: int,
    include_program: bool,
    include_agent_utterance: bool,
    include_described_entities: bool
):
    """Create source contents into a sequence (current utterance, and contexts from previous turns).

    Args:
        turn_lookup (Dict[int, Turn]): [description]
        curr_turn_index (int): [description]
        curr_turn (Turn): [description]
        num_context_turns (int): [description]
        min_turn_index (int): [description]
        include_program (bool): [description]
        include_agent_utterance (bool): [description]
        include_described_entities (bool): [description]
    """
    context_turns = create_context_turns(
            turn_lookup=turn_lookup,
            curr_turn_index=curr_turn_index,
            num_context_turns=num_context_turns,
            min_turn_index=min_turn_index,
            )
    src_str = create_source_str(
        curr_turn=curr_turn,
        context_turns=context_turns,
        include_program=include_program,
        include_agent_utterance=include_agent_utterance,
        include_described_entities=include_described_entities,
        tokenize_utterance=False,
        )
    src_tok_str = create_source_str(
        curr_turn=curr_turn,
        context_turns=context_turns,
        include_program=include_program,
        include_agent_utterance=include_agent_utterance,
        include_described_entities=include_described_entities,
        tokenize_utterance=True,
        )

    src_str_tokens = src_str.split()
    src_tok_str_tokens = src_tok_str.split()

    return src_str_tokens, src_tok_str_tokens


def create_oracle_for_dialogue(
    dialogue: Dialogue,
    program_graph_methods: ProgramGraph,
    *,
    graph_order: str,
    min_turn_index: int,
    num_context_turns: int,
    include_program: bool,
    include_agent_utterance: bool,
    include_described_entities: bool
    ) -> Iterator[Tuple[List[str], List[str]]]:
    """Create the oracle graph action sequences for a Dialogue."""
    turn_lookup: Dict[int, Turn] = {turn.turn_index: turn for turn in dialogue.turns}
    for turn_index, turn in turn_lookup.items():
        if turn.skip:
            continue

        src_str_tokens, src_tok_str_tokens = create_source_contents_for_turn(
            turn_lookup=turn_lookup,
            curr_turn_index=turn_index,
            curr_turn=turn,
            num_context_turns=num_context_turns,
            min_turn_index=min_turn_index,
            include_program=include_program,
            include_agent_utterance=include_agent_utterance,
            include_described_entities=include_described_entities
            )

        # # TODO check tokenization and have string tokens appear the same as in the program
        # src_contents = src_tok_str_tokens

        # use our own tokenizer to be consistent with the target graph strings
        src_contents = tokenize(' '.join(src_str_tokens))

        # try:
        tgt_actions, graph = create_graph_actions_from_dialogue_turn(turn, program_graph_methods, graph_order)
        # except:
        #     print(turn.lispress)
        #     breakpoint()

        tgt_lispress = turn.lispress    # untokenized lispress; a string

        # TODO save tokenized lispress as well
        # breakpoint()

        yield (src_contents, tgt_actions, tgt_lispress)


def run_oracle(
    dataflow_dialogues_jsonl: str,
    out_actions: str,
    out_source: str,
    *,
    out_lispress: str = None,
    graph_order: str = None,
    dataset: str = 'smcalflow',
    **kwargs,
    ) -> None:
    """Create the oracle graph action sequences for a dialogue dataset."""
    src_contents_all = []
    tgt_actions_all = []
    tgt_lispress_all = []

    program_graph_methods = ProgramGraph(type_args_coversion='atomic', dataset=dataset)

    for line in tqdm(open(dataflow_dialogues_jsonl), unit=" dialogues"):
        dialogue: Dialogue
        dialogue = jsons.loads(line.strip(), Dialogue)
        for src_contents, tgt_actions, tgt_lispress in create_oracle_for_dialogue(
            dialogue,
            program_graph_methods,
            graph_order=graph_order,
            **kwargs
        ):
            src_contents_all.append(src_contents)
            tgt_actions_all.append(tgt_actions)
            tgt_lispress_all.append(tgt_lispress)    # untokenized lispress; a string

    # average length
    tgt_actions_avg_len = sum(len(x) for x in tgt_actions_all) / len(tgt_actions_all)
    print(f'Average target sequence length: {tgt_actions_avg_len:.2f}')

    # write to files
    write_tokenized_sentences(
        out_source,
        src_contents_all,
        ' '    # or '\t'
    )

    write_tokenized_sentences(
        out_actions,
        tgt_actions_all,
        ' '    # or '\t'.
    )
    # NOTE If using whitespace ' ', make sure the actions do not include any whitespace, e.g. the arc actions
    #      in the parathesis

    if out_lispress is not None:
        write_string_sentences(
            out_lispress,
            tgt_lispress_all
        )

    return


def parse_args():
    parser = argparse.ArgumentParser(description='Generate oracle graph sequences for SMCalFlow dialogues')
    parser.add_argument('--dialogues_jsonl', type=str,
                        help='the jsonl file containing the dialogue data with dataflow programs')
    parser.add_argument('--out_actions', type=str,
                        help='output file to store the processed target programs as graph actions')
    parser.add_argument('--out_source', type=str,
                        help='output file to store the processed source tokens (including the turn contexts)')
    parser.add_argument('--out_lispress', type=str,
                        help='extract gold lispress from the original dialogue dataset')
    parser.add_argument('--graph_order', type=str, default=None,
                        help='graph generation order (node order) to create the actions')
    parser.add_argument('--dataset', type=str, default='smcalflow', choices=['smcalflow', 'treedst'],
                        help='dataset that we are dealing with')
    parser.add_argument(
        "--num_context_turns",
        type=int,
        help="number of previous turns to be included in the source sequence",
    )
    parser.add_argument(
        "--include_program",
        default=False,
        action="store_true",
        help="if True, include the gold program for the context turn parts",
    )
    parser.add_argument(
        "--include_agent_utterance",
        default=False,
        action="store_true",
        help="if True, include the gold agent utterance for the context turn parts",
    )
    parser.add_argument(
        "--include_described_entities",
        default=False,
        action="store_true",
        help="if True, include the described entities field for the context turn parts",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    run_oracle(args.dialogues_jsonl,
               out_actions=args.out_actions,
               out_source=args.out_source,
               out_lispress=args.out_lispress,
               graph_order=args.graph_order,
               dataset=args.dataset,
               min_turn_index=_MIN_TURN_INDEX,
               num_context_turns=args.num_context_turns,
               include_program=args.include_program,
               include_agent_utterance=args.include_agent_utterance,
               include_described_entities=args.include_described_entities,
               )

    print(f'unmatched lispress after the roundtrip: {count}')
