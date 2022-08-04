# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""API calls definition and approximate run time."""
from typing import Tuple


class CalflowApis:
    def __init__(self, execution_time=100.0):
        self.execution_time = execution_time
        # name, median (ms)
        self.names_and_latencies: Tuple[Tuple[str, float], ...] = (
            ('Yield', execution_time),
            ('RecipientAvailability', execution_time),
            ('FindReports', execution_time),
            ('FindManager', execution_time),
            ('UpdatePreflightEventWrapper', execution_time),
            ('CreatePreflightEventWrapper', execution_time),
            ('DeletePreflightEventWrapper', execution_time),
            ('FindEventWrapperWithDefaults', execution_time),
            ('RecipientWithNameLike', execution_time),
            ('DeleteCommitEventWrapper', execution_time),
            ('UpdateCommitEventWrapper', execution_time),
            ('CreateCommitEventWrapper', execution_time),
            ('EventAttendance', execution_time),
        )
        self.names: Tuple[str, ...] = tuple(name for name, _ in self.names_and_latencies)
        self.latencies: Tuple[float, ...] = tuple(latency for _, latency in self.names_and_latencies)


class TreedstApis:
    def __init__(self, execution_time=100.0):
        self.execution_time = execution_time
        # name, median (ms)
        self.names_and_latencies: Tuple[Tuple[str, float], ...] = (
            ('plan', execution_time),
            ('Create', execution_time),
            ('Find', execution_time),
            ('Update', execution_time),
            ('Delete', execution_time),
            ('Book', execution_time),
            ('CheckExistence', execution_time),
            ('reference', execution_time),
            ('revise', execution_time),
            ('refer', execution_time),
            ('someSalient', execution_time),
        )
        self.names: Tuple[str, ...] = tuple(name for name, _ in self.names_and_latencies)
        self.latencies: Tuple[float, ...] = tuple(latency for _, latency in self.names_and_latencies)
