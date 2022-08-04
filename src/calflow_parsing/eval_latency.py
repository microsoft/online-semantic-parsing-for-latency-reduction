# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Example usage:

    python ./code/src/calflow_parsing/eval_latency.py \
      --predictions-path=path_to_predictions \
      --gold-lispress-path=path_to_gold_lispress \
      --threshold=-1 \
      --output-file=out.jsonl \
      --max-turns=2
"""
import argparse
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from operator import itemgetter as ig
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Iterator, Set, Container

import numpy as np
from tqdm import tqdm

from calflow_parsing.api_constants import CalflowApis, TreedstApis
from calflow_parsing.apis import extract_api_subgraphs
from calflow_parsing.calflow_graph import graph_to_lispress, lispress_to_graph
from calflow_parsing.calflow_machine import CalFlowMachine
from calflow_parsing.conf import DATA_DIR, SAVE_DIR
from calflow_parsing.graph import Graph
from calflow_parsing.word_rate import AsrWord, AsrSentence, PER_SEC, simulate_asr_timings


PREDICTIONS_DIR = (
    SAVE_DIR /
    'exp_treedst_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0' /
    'models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2'
    )
PREDICTIONS_PATH = PREDICTIONS_DIR / "beam1" / 'valid-prefixallt.hypos.json'
GOLD_LISPRESS_PATH = DATA_DIR / 'processed' / 'treedst' / 'act-top_src-ct1-npwa_tgt-cp-str' / 'oracle' / "valid.lispress"

TurnIdx = str
PrefixLen = int


MINIMUM_SCORE = 0    # global variable to record the minimum score


# def graph_to_lispress(graph: Graph) -> str:
#     try:
#         lispress = graph_to_lispress(graph)
#     except RecursionError:
#         lispress = _CONVERSION_ERROR
#     return lispress


# sys.setrecursionlimit(500)


def is_graph_subset(graph_set_a: Set[Graph], graph_set_b: Set[Graph]) -> bool:
    for g in graph_set_a:
        if not is_graph_in(g, graph_set_b):
            return False
    return True


def is_graph_in(graph: Graph, graph_set: Set[Graph]) -> bool:
    for g in graph_set:
        if graph.is_identical(g):
            return True
    return False


# @dataclass(frozen=True)
@dataclass
class ApiCall:
    graph: Graph
    # all times in 100 nanoseconds from start of utterance
    start_time: int
    finish_time: int
    _lispress: str = None

    @property
    def lispress(self):
        if self._lispress is not None:
            return self._lispress
        self._lispress = graph_to_lispress(self.graph)
        return self._lispress

    def to_json(self) -> Dict:
        return {
            'lispress': self.lispress,        # NOTE when dumping results to file, we still dump the converted lispress
            'start_time': self.start_time,
            'finish_time': self.finish_time,
        }


@dataclass(frozen=True)
class ScoredGraph:
    graph: Graph
    node_scores: Dict[int, float]
    arc_scores: Dict[Tuple[int, str, int], float]


@dataclass(frozen=True)
class BeamItem:
    scored_actions: List[Tuple[str, float]]

    @property
    def actions(self) -> List[str]:
        return [a for a, _ in self.scored_actions]

    @property
    def scores(self) -> List[float]:
        return [s for _, s in self.scored_actions]

    def to_graph(self) -> ScoredGraph:
        machine = CalFlowMachine()
        machine.apply_actions(self.actions)
        node_scores = {}
        arc_scores = {}
        for maybe_node, maybe_edge, score in zip(
                machine.action_graph_nodes,
                machine.action_graph_edges,
                self.scores,
        ):
            if maybe_node is not None:
                node_scores[maybe_node] = score
            if maybe_edge is not None:
                arc_scores[maybe_edge] = score
        return ScoredGraph(
            graph=machine.graph,
            node_scores=node_scores,
            arc_scores=arc_scores,
        )


@dataclass(frozen=True)
class Prediction:
    beam: List[BeamItem]


@dataclass(frozen=True)
class PolicyState:
    time: int
    prefix: List[AsrWord]
    returned_calls: Set[Graph]
    in_flight_calls: List[ApiCall]
    API_P50_DICT: Dict[str, int]

    def is_ready(self, graph: Graph) -> bool:
        dependents = set(extract_api_subgraphs(graph, names=tuple(self.API_P50_DICT.keys())))

        # don't count ourself as a dependent
        for g in dependents:
            if g.is_identical(graph):
                dependents.remove(g)
                break

        # we're ready if all our dependents have completed
        return is_graph_subset(dependents, self.returned_calls)

    def make_all_possible_calls(self, graphs: List[Graph]) -> Tuple["PolicyState", List[Graph]]:
        """
        Returns a new state, with each `g` in `graphs` added to
        `in_flight_calls` iff all of its dependents have completed.
        Also returns the list of graphs that could not be immediately called.
        """
        api_calls: List[ApiCall] = []
        unready_graphs: List[Graph] = []
        launched_calls = self.returned_calls | {call.graph for call in self.in_flight_calls}
        for graph in graphs:
            # don't waste time on calls that we've already launched
            if not is_graph_in(graph, launched_calls):
                if self.is_ready(graph):
                    api_calls.append(
                        ApiCall(
                            graph=graph,
                            start_time=self.time,
                            finish_time=self.time + self.API_P50_DICT[graph.nodes[graph.root]],
                        )
                    )
                    launched_calls.add(graph)
                else:
                    unready_graphs.append(graph)
        in_flight_calls = sorted(self.in_flight_calls + api_calls, key=lambda a: a.finish_time)
        return replace(self, in_flight_calls=in_flight_calls), unready_graphs

    def complete_and_pop_in_flight_call(self) -> Tuple[ApiCall, "PolicyState"]:
        """
        Returns the next API call to finish, along with the the new state
        after that call completes.
        """
        if not self.in_flight_calls:
            raise StopIteration
        first_call, *rest = self.in_flight_calls
        state = replace(
            self,
            time=first_call.finish_time,
            returned_calls=(self.returned_calls | {first_call.graph}
                            if not is_graph_in(first_call.graph, self.returned_calls)
                            else
                            self.returned_calls),
            in_flight_calls=rest,
        )
        return first_call, state


class Policy(ABC):
    """
    A policy decides which API calls to make, conditioned on a model's
    prediction and the current execution state.
    """
    @abstractmethod
    def suggest_api_calls(self, state: PolicyState, prediction: Prediction) -> List[Graph]:
        raise NotImplementedError()


@dataclass(frozen=True)
class WaitUntilEndPolicy(Policy):
    """Never makes an API call until the utterance is complete."""
    api_names: Container[str]

    def suggest_api_calls(self, state: PolicyState, prediction: Prediction) -> List[Graph]:
        return []


@dataclass(frozen=True)
class LogProbThresholdPolicy(Policy):
    """
    Makes an API call if it's log probability is above `threshold`.
    Currently defining the logprob of a subgraph to be the min logprob of any
    node or arc in the subgraph, but could try sum, avg, or other things.
    """
    threshold: float
    api_names: Container[str]

    @staticmethod
    def _score_subgraph(subgraph: Graph, scored_graph: ScoredGraph) -> float:
        if not subgraph.nodes:
            return 0.0
        node_scores = [scored_graph.node_scores[n] for n in subgraph.nodes]
        arc_scores = [scored_graph.arc_scores[a] for a in subgraph.edges]
        # TODO: sum? average?
        # TODO: consider scores outside of the subgraph?
        return min(node_scores + arc_scores)
        # return sum(node_scores + arc_scores)

    def suggest_api_calls(self, state: PolicyState, prediction: Prediction) -> List[Graph]:
        scored_graphs = (item.to_graph() for item in prediction.beam)

        # policy to score the subgraph with action-level logprobs
        scored_api_subgraphs = [
            (subgraph, self._score_subgraph(subgraph, g))
            for g in scored_graphs
            for subgraph in extract_api_subgraphs(g.graph, names=self.api_names)
            if (state.is_ready(subgraph)
                and not is_graph_in(subgraph, state.returned_calls))
        ]

        # # ===== try another policy: use the beam score for all subgraphs
        # beam_scores = (sum([sa[1] for sa in item.scored_actions]) for item in prediction.beam)
        # scored_api_subgraphs = [
        #     (subgraph, bs)
        #     for g, bs in zip(scored_graphs, beam_scores)
        #     for subgraph in extract_api_subgraphs(g.graph, names=self.api_names)
        #     if (state.is_ready(subgraph)
        #         and not is_graph_in(subgraph, state.returned_calls))
        # ]
        # # ==========

        global MINIMUM_SCORE
        all_scores = [score for subgraph, score in scored_api_subgraphs]
        if all_scores:
            if (min_score := min(all_scores)) < MINIMUM_SCORE:
                MINIMUM_SCORE = min_score

        above_threshold = [
            (subgraph, score)
            for subgraph, score in scored_api_subgraphs
            if score >= self.threshold
        ]
        above_threshold.sort(key=ig(1), reverse=True)
        return [g for g, _ in above_threshold]


def drive_policy(
        policy: Policy,
        predictions: Dict[PrefixLen, Prediction],
        # TODO: `utterance: Dict[PrefixLen, AsrSentence]`
        #  to allow for partial ASR results that don't match the final ASR results
        utterance: AsrSentence,
        final_prediction: Optional[Graph],
        # TODO: add max_in_flight_budget
        API_P50_DICT: Dict[str, int],
) -> Iterator[ApiCall]:
    """
    Returns a list of completed API calls (each with a non-None finish_time).
    Re-runs the policy every time an utterance token ends, or an in_flight_call returns.
    """
    assert utterance.words
    n = len(utterance.words) + 1
    assert set(predictions.keys()) == set(range(n))

    final_graphs = extract_api_subgraphs(final_prediction, names=policy.api_names) if final_prediction is not None else []

    # translate all timings so that end of utterance is at t=0
    finish = utterance.words[-1].end
    words = [
        replace(word, offset=word.offset - finish)
        for word in utterance.words
    ]

    start = words[0].offset
    word_ends = [(0, start)] + [
        (i + 1, word.end)
        for i, word in enumerate(words)
    ]
    prefix_len = 0
    state: PolicyState = PolicyState(
        time=start,
        prefix=[],
        returned_calls=set(),
        in_flight_calls=[],
        API_P50_DICT=API_P50_DICT,
    )
    for prefix_len, time in word_ends:
        # run policy when api calls complete before token ends
        while state.in_flight_calls and state.in_flight_calls[0].finish_time < time:
            first_call, state = state.complete_and_pop_in_flight_call()
            yield first_call
            if prefix_len > 0:
                api_calls = policy.suggest_api_calls(state=state, prediction=predictions[prefix_len - 1])
                state, _ = state.make_all_possible_calls(api_calls)

        # run policy when token ends
        state = replace(state, time=time, prefix=words[:prefix_len])
        api_calls = policy.suggest_api_calls(state=state, prediction=predictions[prefix_len])
        state, _ = state.make_all_possible_calls(api_calls)

    # now that utterance has finished, do calls needed for final_prediction as fast as possible
    state, final_graphs = state.make_all_possible_calls(final_graphs)

    # yield all calls that finish after the utterance ends
    while state.in_flight_calls:
        first_call, state = state.complete_and_pop_in_flight_call()
        yield first_call
        api_calls = policy.suggest_api_calls(state=state, prediction=predictions[prefix_len])
        state, final_graphs = state.make_all_possible_calls(api_calls + final_graphs)

    assert not final_graphs, [graph_to_lispress(g) for g in final_graphs]


def evaluate_policy_predictions(
        predictions: List[ApiCall],
        gold: Graph,
        api_names: Container[str],
) -> int:
    """
    Returns the number of 100 nanoseconds after the utterance ends that the final gold API call returns.
    Re-runs the policy every time an utterance token ends, or an in_flight_call returns.
    """
    # gold_api_lispresses = {
    #     graph_to_lispress(g)
    #     for g in extract_api_subgraphs(gold, names=api_names)
    # }

    # completed_api_lispresses = {}
    # for api_call in predictions:
    #     if api_call.lispress not in completed_api_lispresses:
    #         completed_api_lispresses[api_call.lispress] = api_call.finish_time
    #     else:
    #         # NOTE only record the earliest time for each API call, as there are many in the list
    #         if api_call.finish_time < completed_api_lispresses[api_call.lispress]:
    #             completed_api_lispresses[api_call.lispress] = api_call.finish_time

    # # TODO: currently `drive_policy` is responsible for making sure all
    # #  `gold_api_lispresses` appear in `completed_api_lispresses`.
    # #  Might should be our responsibility here.
    # finish_times = [completed_api_lispresses[gold_api_lispress] for gold_api_lispress in gold_api_lispresses]

    gold_api_graphs = {g for g in extract_api_subgraphs(gold, names=api_names)}
    completed_api_graphs = {api_call.graph: api_call.finish_time for api_call in predictions}
    # TODO: currently `drive_policy` is responsible for making sure all
    #  `gold_api_lispresses` appear in `completed_api_lispresses`.
    #  Might should be our responsibility here.
    # NOTE we directly use the graph to compare instead of lispress, as graph -> lispress might not be 1-to-1.
    finish_times = []
    for gold_api_graph in gold_api_graphs:
        for completed_api_graph, finish_time in completed_api_graphs.items():
            if gold_api_graph.is_identical(completed_api_graph):
                finish_times.append(finish_time)
                break
        else:
            raise ValueError('Gold API graph is not in completed API graphs')

    return max(finish_times) if finish_times else 0.0


def load_predictions(path: Path = PREDICTIONS_PATH) -> Tuple[Dict[TurnIdx, Dict[PrefixLen, Prediction]], Dict[TurnIdx, List[str]]]:
    with open(path, mode='r', encoding='utf8') as predictions_file:
        predictions_json = json.load(predictions_file)
    all_predictions = {
        turn_idx: {
            int(prefix_len): Prediction([
                BeamItem(list(zip(
                    beam_item['actions'],
                    beam_item['positional_scores'],
                )))
                for beam_item in beam_items
            ])
            for prefix_len, beam_items in preds['predictions'].items()
        }
        for turn_idx, preds in predictions_json.items()
    }
    utterances = {
        turn_idx: predictions['utterance'].split()
        for turn_idx, predictions in predictions_json.items()
    }
    return all_predictions, utterances


def load_gold_lispress(path: Path = GOLD_LISPRESS_PATH) -> Dict[TurnIdx, str]:
    with open(path, mode='r', encoding='utf8') as gold_lispress_file:
        return {
            str(i): line.strip()
            for i, line in enumerate(gold_lispress_file)
        }


@dataclass(frozen=True)
class Stats:
    # in seconds
    latency: float
    api_calls: List[ApiCall]


def simulate_and_evaluate(
        all_predictions: Dict[TurnIdx, Dict[PrefixLen, Prediction]],
        utterances: Dict[TurnIdx, List[str]],
        gold: Dict[TurnIdx, str],
        policies: Tuple[Policy, ...] = (LogProbThresholdPolicy(-1, api_names=CalflowApis().names),
                                        WaitUntilEndPolicy(api_names=CalflowApis().names)),
        word_rate_model: str = 'char-linear',
        API_P50_DICT: Dict[str, int] = None,
        quiet: bool = True,
) -> List[Dict[Policy, Stats]]:
    results = []
    for idx, predictions in tqdm(all_predictions.items()):
        utterance = utterances[idx]
        gold_lispress = gold[idx]
        if not quiet:
            print()
            print(' '.join(utterance))
            print(gold_lispress)
        if word_rate_model == 'char-linear':
            sentence = AsrSentence(list(simulate_asr_timings(utterance)))
        elif word_rate_model == 'constant':
            sentence = AsrSentence(list(simulate_asr_timings(utterance, slope=0, intercept=1)))
            # 1 unit is 1s/1000ms -> set the exectution time accordingly (1000 is the time spent for a word)
        else:
            raise NotImplementedError
        gold_graph = lispress_to_graph(gold_lispress)
        result = {}
        for policy in policies:
            api_calls = list(drive_policy(
                policy=policy,
                predictions=predictions,
                utterance=sentence,
                final_prediction=gold_graph,
                API_P50_DICT=API_P50_DICT,
            ))
            latency = evaluate_policy_predictions(
                predictions=api_calls,
                gold=gold_graph,
                api_names=tuple(API_P50_DICT.keys()),
            )
            result[policy] = Stats(
                latency=latency / PER_SEC,
                api_calls=api_calls,
            )
            if not quiet:
                print()
                print(policy)
                print('latency', latency / PER_SEC)
                print('api_calls')
                for api_call in api_calls:
                    print(api_call)
            # record information about gold lispress, utterance and timing
            result['gold_lispress'] = gold_lispress
            result['utterance'] = utterance    # List[str]
            result['utterance_timings'] = [asrw.end for asrw in sentence.words]
            # List[int], timings are in units of 100 nanoseconds
        results.append(result)
    return results


def summarize_stats(all_stats: List[Dict[Policy, Stats]]) -> Tuple[float, float]:
    keys = list(all_stats[0].keys())
    baseline, = [k for k in keys if isinstance(k, WaitUntilEndPolicy)]
    other, = [k for k in keys if isinstance(k, LogProbThresholdPolicy)]

    def latency_improvement(stats_dict: Dict[Policy, Stats]) -> float:
        baseline_latency = max(stats_dict[baseline].latency, 0)
        other_latency = max(stats_dict[other].latency, 0)
        return baseline_latency - other_latency

    def latency_improvement_nocap(stats_dict: Dict[Policy, Stats]) -> float:
        baseline_latency = stats_dict[baseline].latency
        other_latency = stats_dict[other].latency
        return baseline_latency - other_latency

    def latency_improvement_oracle(stats_dict: Dict) -> float:
        utterance_end = stats_dict['utterance_timings'][-1] / PER_SEC
        baseline_latency = stats_dict[baseline].latency
        assert baseline_latency >= 0
        oracle = min(utterance_end, baseline_latency)
        return oracle

    avg_latency_improvement = np.mean([latency_improvement(d) for d in all_stats])
    avg_num_api_calls = np.mean([len(d[other].api_calls) for d in all_stats])
    avg_baseline_latency = np.mean([d[baseline].latency for d in all_stats])
    avg_num_baseline_api_calls = np.mean([len(d[baseline].api_calls) for d in all_stats])

    avg_latency_improvement_oracle = np.mean([latency_improvement_oracle(d) for d in all_stats])

    avg_latency_improvement_nocap = np.mean([latency_improvement_nocap(d) for d in all_stats])
    avg_latency_nocap = np.mean([d[other].latency for d in all_stats])
    print()
    print()
    print("avg_latency_improvement:", avg_latency_improvement)
    print("avg_num_api_calls:", avg_num_api_calls)
    print("avg_WaitUntilEnd_latency:", avg_baseline_latency)
    print("avg_num_WaitUntilEnd_api_calls:", avg_num_baseline_api_calls)
    print("avg_latency_improvement_oracle:", avg_latency_improvement_oracle)
    return (avg_latency_improvement, avg_num_api_calls, avg_baseline_latency, avg_num_baseline_api_calls,
            avg_latency_improvement_nocap, avg_latency_nocap, avg_latency_improvement_oracle)


def main(
        predictions_path: Path = PREDICTIONS_PATH,
        gold_lispress_path: Path = GOLD_LISPRESS_PATH,
        output_file: Path = "out.jsonl",
        threshold: float = -1.0,
        max_turns: Optional[int] = None,
        word_rate_model: str = 'char-linear',
        API_P50_DICT: Dict[str, int] = None,
        execution_time: float = None,
        quiet: bool = True,
):

    ps, us = load_predictions(predictions_path)
    if max_turns is not None:
        ps = dict(list(ps.items())[:max_turns])
    gls = load_gold_lispress(gold_lispress_path)
    # debug after data loading (only works by directly calling the python script instead of wrapping in bash script)
    # breakpoint()
    api_names = tuple(API_P50_DICT.keys())
    policies = (
        LogProbThresholdPolicy(threshold, api_names=api_names),
        WaitUntilEndPolicy(api_names=api_names)
        )
    results = simulate_and_evaluate(
        all_predictions=ps,
        utterances=us,
        gold=gls,
        policies=policies,
        word_rate_model=word_rate_model,
        API_P50_DICT=API_P50_DICT,
    )

    with open(output_file, 'w') as out_file:
        for result in results:
            # policy results
            js = {
                str(k): {
                    'latency': v.latency,
                    'api_calls': [c.to_json() for c in v.api_calls],
                }
                for k, v in result.items() if 'Policy' in str(k)
            }
            # general information: utterance and timings, gold lispress
            for k, v in result.items():
                if 'Policy' in str(k):
                    continue
                js[k] = v

            out_file.write(json.dumps(js) + "\n")

    summary = summarize_stats(results)
    avg_latency_improvement, avg_num_api_calls, avg_baseline_latency, avg_num_baseline_api_calls, \
        avg_latency_improvement_nocap, avg_latency_nocap, avg_latency_improvement_oracle = summary
    if output_file is not None:
        with open(str(output_file) + ".summary", 'w') as summary_file:
            global MINIMUM_SCORE
            summary_file.write("MINIMUM_SCORE: " + str(MINIMUM_SCORE) + "\n")
            summary_file.write("WORD_RATE_MODEL: " + word_rate_model + "\n")
            if execution_time is not None:
                summary_file.write("EXECUTION_TIME: " + str(execution_time) + "\n")
            summary_file.write("threshold: " + str(threshold) + "\n")
            summary_file.write("avg_latency_improvement: " + str(avg_latency_improvement) + "\n")
            summary_file.write("avg_num_api_calls: " + str(avg_num_api_calls) + "\n")
            summary_file.write("avg_WaitUntilEnd_latency: " + str(avg_baseline_latency) + "\n")
            summary_file.write("avg_num_WaitUntilEnd_api_calls: " + str(avg_num_baseline_api_calls) + "\n")
            summary_file.write("avg_latency_improvement_oracle: " + str(avg_latency_improvement_oracle) + "\n")
            summary_file.write("avg_latency_improvement_nocap: " + str(avg_latency_improvement_nocap) + "\n")
            summary_file.write("avg_latency_nocap: " + str(avg_latency_nocap) + "\n")
    return summary, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-path",
        type=str,
        help="json file with model predictions",
        default=PREDICTIONS_PATH,
    )
    parser.add_argument(
        "--gold-lispress-path",
        type=str,
        help="text file with gold lispress, one per line",
        default=GOLD_LISPRESS_PATH,
    )
    parser.add_argument(
        '--word-rate-model',
        type=str,
        help='word speaking rate scheme (when "constant", 1000 in "--execution-time" is 1 unit)',
        choices=['constant', 'char-linear', 'real-voice'],    # 'constant' is the intrinsic setting
        default='char-linear',
    )
    parser.add_argument(
        '--execution-time',
        type=float,
        help='execution time of a function node (in millisecond)',
        default=100,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="logprob threshold to make api calls",
        default=-1.0,
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="truncate to the first `max_turns` turns if given",
        default=None,
    )
    parser.add_argument(
        "--quiet",
        type=int,
        help="whether to display on screen details of each example running",
        default=1,
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="where to write results",
        default='tmp.lay',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='smcalflow',
        choices=['smcalflow', 'treedst'],
        help='dataset that we are dealing with'
    )

    args = parser.parse_args()

    # convert from millis to 100 nanos
    API_P50_DICT = {
        name: int(ms * PER_SEC / 1000)
        for name, ms in CalflowApis(args.execution_time).names_and_latencies
    }
    if args.dataset == 'treedst':
        API_P50_DICT = {
            name: int(ms * PER_SEC / 1000)
            for name, ms in TreedstApis(args.execution_time).names_and_latencies
        }

    main(
        predictions_path=Path(args.predictions_path).resolve(),
        gold_lispress_path=Path(args.gold_lispress_path).resolve(),
        output_file=Path(args.output_file).resolve(),
        threshold=args.threshold,
        max_turns=args.max_turns,
        word_rate_model=args.word_rate_model,
        API_P50_DICT=API_P50_DICT,
        execution_time=args.execution_time,    # to write in the output summary file
        quiet=bool(args.quiet),
    )
