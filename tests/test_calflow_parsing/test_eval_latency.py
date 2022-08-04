# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
from typing import Dict, List

from calflow_parsing.calflow_graph import ProgramGraph
from calflow_parsing.calflow_machine import CalFlowOracle
from calflow_parsing.eval_latency import drive_policy, LogProbThresholdPolicy, PrefixLen, Prediction, \
    evaluate_policy_predictions, BeamItem, WaitUntilEndPolicy, ApiCall
from calflow_parsing.word_rate import AsrSentence, simulate_asr_timings, PER_SEC

UTTERANCE = ["Can", "you", "make", "an", "event", "for", "fri"]
LISPRESS = """
(Yield
  (CreateCommitEventWrapper
    (CreatePreflightEventWrapper
      (Event.start_? (DateTime.date_? (?= (NextDOW (Friday))))))))
"""


def test_drive_and_eval():
    policy = LogProbThresholdPolicy(math.log(8/9))
    utterance = AsrSentence(list(simulate_asr_timings(UTTERANCE)))
    n = len(UTTERANCE) + 1
    graph = ProgramGraph(type_args_coversion='atomic').lispress_to_graph(LISPRESS)
    actions = CalFlowOracle.graph_to_actions(graph=graph)
    predictions: Dict[PrefixLen, Prediction] = {
        i: Prediction(beam=[BeamItem([(a, math.log((i+1)/n)) for a in actions])])
        for i in range(n)
    }
    final_prediction = graph
    api_calls: List[ApiCall] = list(drive_policy(
        policy=policy,
        predictions=predictions,
        utterance=utterance,
        final_prediction=final_prediction,
    ))
    print()
    print(policy)
    print(len(api_calls), "calls made:")
    for api_call in api_calls:
        print(" ", api_call)
    latency = evaluate_policy_predictions(
        predictions=api_calls,
        gold=final_prediction,
    )
    print("latency from end of utterance:")
    print(" ", latency / PER_SEC, "s")
    # every call must be unique
    assert(len({c.lispress for c in api_calls}) == len(api_calls))

    baseline = WaitUntilEndPolicy()
    api_calls = list(drive_policy(
        policy=baseline,
        predictions=predictions,
        utterance=utterance,
        final_prediction=final_prediction,
    ))
    print()
    print(baseline)
    print(len(api_calls), "calls made:")
    for api_call in api_calls:
        print(" ", api_call)
    latency = evaluate_policy_predictions(
        predictions=api_calls,
        gold=final_prediction,
    )
    print("latency from end of utterance:")
    print(" ", latency / PER_SEC, "s")
