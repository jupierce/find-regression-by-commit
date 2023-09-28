#!/usr/bin/env python3
from __future__ import annotations

import math
import multiprocessing
import datetime
import os
from typing import NamedTuple, List, Dict, Set, Tuple, Optional
from enum import Enum
import re
import pathlib
import functools
import weakref

import click
import numpy
import tqdm
from collections import OrderedDict

from airium import Airium

import itertools
import pandas
from google.cloud import bigquery
from fast_fisher import fast_fisher_cython
from functools import cached_property

pandas.options.compute.use_numexpr = True


def get_prowjob_url(row):
    return row['prowjob_url']

class Mode(Enum):
    PRIORITIZE_NEWEST_TEST_RESULTS = 0  # Useful if we are triggering new tests to evaluate signal or want to see if a signal is falling
    PRIORITIZE_ORIGINAL_TEST_RESULTS = 1


RELEASE_NAME_SPLIT_REGEX = re.compile(r"(.*-0\.[^-]+)-(.*)")


def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)
            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator


class ReleasePayloadStreams(Enum):
    CI_PAYLOAD = 'ci'
    NIGHTLY_PAYLOAD = 'nightly'
    PR_PAYLOAD = 'pr'

    @staticmethod
    def get_stream(release_name: str) -> Optional[ReleasePayloadStreams]:
        if '-0.ci-' in release_name:
            return ReleasePayloadStreams.CI_PAYLOAD

        if '-0.nightly-' in release_name:
            return ReleasePayloadStreams.NIGHTLY_PAYLOAD

        if '-0.ci.test-' in release_name:
            return ReleasePayloadStreams.PR_PAYLOAD

        return None

    @staticmethod
    def split(release_name: str) -> Optional[Tuple[str, str]]:
        """
        Splits a release name into a prefix and suffix.
        :param release_name: e.g. "4.14.0-0.nightly-2023-07-10-065310"
        :return: e.g. ("4.14.0-0.nightly", "2023-07-10-065310")
        """
        m = RELEASE_NAME_SPLIT_REGEX.match(release_name)
        if not m:
            return 'UnknownStream', 'Unknown'
        return m.group(1), m.group(2)


class WrappedInteger:
    def __init__(self, value=0):
        self._value = int(value)

    def inc(self, d=1):
        self._value += int(d)
        return self._value

    def dec(self, d=1):
        return self.inc(-d)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = int(v)
        return self._value


class ProwJobId(NamedTuple):
    prowjob_name: str
    prowjob_build_id: str
    modified_time: datetime.datetime

    def link(self) -> str:
        return f'https://prow.ci.openshift.org/view/gs/origin-ci-test/logs/{self.prowjob_name}/{self.prowjob_build_id}'


class ProwJobRun:
    def __init__(self, commit_id, prowjob_id: ProwJobId):
        self.commit_id = commit_id
        self.id = prowjob_id
        self.success_count = 0
        self.failure_count = 0
        self.flake_count = 0

    def add_success(self):
        self.success_count += 1

    def add_failure(self, amount):
        self.failure_count += amount
        if amount < 0:
            self.flake_count += amount


class OutcomePair:

    def __init__(self, success_count: int, failure_count: int, flake_count: int):
        self.success_count = success_count
        self.failure_count = failure_count
        self.flake_count = flake_count

    def pass_rate(self) -> float:
        if self.success_count > 0 or self.failure_count > 0:
            return self.success_count / (self.success_count + self.failure_count)
        return 0.0

    def fishers_exact(self, result_sum2) -> float:
        # if we have regressed at least 10% relative to the result we are comparing to.
        if result_sum2.pass_rate() > self.pass_rate():
            return -1 + fast_fisher_cython.fisher_exact(
                self.failure_count, self.success_count,
                result_sum2.failure_count, result_sum2.success_count,
                alternative='greater')
        elif self.pass_rate() > result_sum2.pass_rate():
            # If there has been improvement
            return 1 - fast_fisher_cython.fisher_exact(
                self.failure_count, self.success_count,
                result_sum2.failure_count, result_sum2.success_count,
                alternative='less')

        return 0.0

    def mlog10Test1t(self, result_sum2) -> float:
        if result_sum2.pass_rate() > self.pass_rate():
            return -1 * fast_fisher_cython.mlog10Test1t(
                self.failure_count, self.success_count,
                result_sum2.failure_count, result_sum2.success_count
            )
        elif self.pass_rate() > result_sum2.pass_rate():
            # If there has been improvement
            return fast_fisher_cython.mlog10Test1t(
                self.failure_count, self.success_count,
                result_sum2.failure_count, result_sum2.success_count
            )
        return 0.0

    @classmethod
    def outcome_sums(cls, selection: pandas.DataFrame) -> OutcomePair:
        test_runs = len(selection)
        if test_runs > 0:
            success_count = selection['success_val'].values.sum()
            flake_count = selection['flake_count'].values.sum()
        else:
            # No tests have been run against this commit
            success_count = 0
            flake_count = 0

        failure_count = test_runs - success_count - flake_count
        if failure_count < 0:
            # Rare case where the window of our select catches a flake but not the error record
            # it is trying to account for.
            failure_count = 0

        return OutcomePair(success_count=success_count, failure_count=failure_count, flake_count=flake_count)

    def __str__(self):
        return f'[s={self.success_count} f={self.failure_count} r={int(self.pass_rate() * 100)}%]'


MAX_WINDOW_SIZE = 30


class ReleasePayload:

    def __init__(self, release_name: ReleaseName, release_created: pandas.Timestamp):
        self.release_name = release_name
        self.release_created = release_created
        self.commits: Dict[str, 'CommitOutcomes'] = dict()
        self.diff_commits: Set[CommitId] = set()
        self.stream_name = ReleasePayloadStreams.get_stream(self.release_name)
        self.stream_prefix, self.stream_suffix = ReleasePayloadStreams.split(self.release_name)

    def add_commit(self, commit_outcome: 'CommitOutcomes', first_release_in_stream_with_commit):
        """
        :param commit_outcome: Information about the commit in this payload.
        :param first_release_in_stream_with_commit: Whether this payload is the first to have included the commit within
                                                    its respective payload stream (CI xor nightly).
        """
        if commit_outcome.commit_id:
            self.commits[commit_outcome.commit_id] = commit_outcome
            if first_release_in_stream_with_commit:
                self.diff_commits.add(commit_outcome.commit_id)


class CommitOutcomes:

    def __init__(self, source_location: str, commit_id: str, created_at: Optional[pandas.Timestamp]):
        self.source_location = source_location

        self.repo_name = source_location.split('/')[-1] or '__'

        self.commit_id = commit_id
        self.created_at = created_at
        self.parent = None
        self.child = None

        self.repo_test_records: Optional[pandas.DataFrame] = None
        self.index_first_self_test_in_records = 0
        self.index_last_self_test_in_records = 0

        self.discrete_outcomes: Optional[OutcomePair] = None
        self.tests_against_this_commit: Optional[pandas.DataFrame] = None

        self.release_streams: Dict[ReleasePayloadStreams, OrderedDict[ReleaseName, ReleasePayload]] = {
            ReleasePayloadStreams.NIGHTLY_PAYLOAD: OrderedDict(),  # We really just want an Ordered Set. Value doesn't matter.
            ReleasePayloadStreams.CI_PAYLOAD: OrderedDict(),
            ReleasePayloadStreams.PR_PAYLOAD: OrderedDict()
        }

    def set_data(self, repo_test_records: pandas.DataFrame):
        """
        :param repo_test_records: Test records ordered by commit and then test date. Dataframe index must be
                                    monotonically increasing / continuous in order for selection logic to work.
                                    Other methods of selection are possible, but using indices is extremely performant.
        :return: None
        """
        self.repo_test_records = repo_test_records

        self.tests_against_this_commit = self.repo_test_records[self.repo_test_records['commits'] == self.commit_id]
        self.discrete_outcomes = OutcomePair.outcome_sums(self.tests_against_this_commit)

        self.index_first_self_test_in_records = self.tests_against_this_commit.first_valid_index()
        self.index_last_self_test_in_records = self.tests_against_this_commit.last_valid_index()

    def record_observation_in_release(self, release_payload: ReleasePayload):
        """
        Records in this object that this commit was tested as part of a ReleasePayload
        and also informs the ReleasePayload object that this commit was tested as part of the payload.
        :param release_payload: The payload the test was run against.
        """
        first_release_in_stream_with_this_commit = False
        if release_payload.stream_name:
            self.release_streams[release_payload.stream_name][release_payload.release_name] = release_payload
            first_release_in_stream_with_this_commit = len(self.release_streams[release_payload.stream_name]) == 1

        release_payload.add_commit(self, first_release_in_stream_with_this_commit)

    def get_ancestor_ids(self) -> List[str]:
        ancestors_commit_ids = []
        node = self.parent
        while node:
            ancestors_commit_ids.append(node.commit_id)
            node = node.parent
        return ancestors_commit_ids

    def get_descendant_ids(self) -> List[str]:
        descendant_ids = []
        node = self.child
        while node:
            descendant_ids.append(node.commit_id)
            node = node.child
        return descendant_ids

    def get_parent_commit_id(self):
        if self.parent:
            return self.parent.commit_id
        return None

    def get_child_commit_id(self):
        if self.child:
            return self.child.commit_id
        return None

    def ahead(self, count: int, mode: Mode = Mode.PRIORITIZE_ORIGINAL_TEST_RESULTS, after_date: Optional[numpy.datetime64] = None) -> pandas.DataFrame:
        outcomes: Optional[pandas.DataFrame] = None

        # if self.index_last_self_test_in_records - self.index_first_self_test_in_records >= count:
        #     return self.tests_against_this_commit

        if after_date:
            # If date filtering is being used, the caller cares about specific time periods, so we
            # use ahead_records_chron. Unlike Mode.PRIORITIZE_NEWEST_TEST_RESULTS ahead calls, this means we will not
            # prefer new test results over old.
            ahead_records_chron = self.repo_test_records
            outcomes = ahead_records_chron[(ahead_records_chron.index >= self.index_first_self_test_in_records) & (ahead_records_chron['modified_time'] >= after_date)]
            if count > -1:
                found_count = len(outcomes)
                if found_count >= count:
                    outcomes = outcomes.iloc[:count]  # Return closest after date (ignore mode since it would always return newest results)
                else:
                    # Not enough results after date, so slide collection window back as far as we can from that date
                    first_index_of_desired_date = outcomes.first_valid_index()  # Where did we start finding data?
                    if first_index_of_desired_date is None:
                        # There are no records for this commit's (or its ancestors) future test results.
                        # Just return the oldest records we have.
                        outcomes = ahead_records_chron[self.index_first_self_test_in_records:][-1 * count:]
                    else:
                        remaining = count - found_count
                        backup_to = first_index_of_desired_date - remaining
                        if backup_to < self.index_first_self_test_in_records:
                            # If we backup to collect the required amount, we would return records before our commit was introduced.
                            # So limit how far we backup.
                            backup_to = self.index_first_self_test_in_records
                        outcomes = ahead_records_chron.iloc[backup_to:backup_to+count]

        if outcomes is None:
            if mode == Mode.PRIORITIZE_NEWEST_TEST_RESULTS:
                # Get the most recent results for the commit possible
                # Try to back up <count> from the last time we see this commit in the test records. If this takes us
                # beyond the first time we see the commit, stop.
                starting_index = max(self.index_last_self_test_in_records - count, self.index_first_self_test_in_records)
            else:
                # Otherwise, our starting index is the earliest modified_time test for the commit.
                starting_index = self.index_first_self_test_in_records

            outcomes = self.repo_test_records.iloc[starting_index:starting_index + count]

        return outcomes

    def ahead_outcome(self, count: int, mode: Mode = Mode.PRIORITIZE_ORIGINAL_TEST_RESULTS, after_date: Optional[numpy.datetime64] = None) -> OutcomePair:
        return OutcomePair.outcome_sums(self.ahead(count=count, mode=mode, after_date=after_date))

    # Consider a list of test results ordered first by commit history
    # (in our test results, commit order is only guaranteed for a given
    # repo, so in the following example, A -> B -> C are all from the
    # same repo) and then by time at which the test run was observed:
    #    commitA
    #       test result tA+0
    #       test result tA+1
    #       ...
    #       test result tA+M
    #    commitB
    #       test result tB+0
    #       test result tB+1
    #       ...
    #       test result tB+N
    #    commitC
    #       test result tC+0
    #       test result tC+1
    #       ...
    #       test result tC+O
    #
    # Note that since tests can occur for a given commit at any time in our scan window,
    # there is no guarantee that, for example, all tA times are greater than tB times.
    #
    # Assuming that we iterate forward through these results to analyze fe10 test successes
    # and failures that happened before and after a commit. This approach has drawbacks.
    #
    # 1. When informing self about successes/failures ahead, we want to recognize the
    #    results of tB+N for commitB because it represents the most recent test
    #    results. This is important because if we devise a system that triggers
    #    test runs to help ferret out real regressions vs likely regression commits
    #    then these will happen later in chronological time. We want to give
    #    then priority so that they impact our FE values instead of being ignored
    #    because we already had fe30 accumulators filled up from tB+0, ....
    # 2. When informing ancestors about successes/failures ahead, we also want
    #    to include the latest test runs first -- for the same reason.
    # 3. When informing children of success, the above ordering would lead commitA's
    #    test runs to be contributed to commitC's behind accumulator before commitB's.
    #    Obviously, the results of the parent commitB are more pertinent to commitC,
    #    so we need to inform children by processing commits in reverse order of
    #    commit history. As with (1) and (2), newer results should be preferred.
    #
    # So should we instead always prefer the newest test records? This also has drawbacks.
    # If the newest records are preferred and the last test records associated with a commit
    # happen to be in payloads including a regression unrelated to the commit, then
    # the commit will be flagged as a regression despite the possibility that it ran
    # the same test successfully for a long period prior to the introduction of the regression.
    #
    # To balance these drawbacks, ahead_outcome_best(...) will select the best fe10 between
    # the newest and oldest tests runs associated with a commit.
    # 1. If test runs are being triggered after a regression signal to isolate whether
    #    regression is real, if a commit did not introduce the regression, then the _best
    #    outcome will eventually gravitate to the new, successful runs.
    # 2. If a commit has numerous successful runs before being included in a payload with
    #    a commit causing a regression, then the earlier test records will be preferred
    #    by _best.

    @memoized_method(maxsize=10)
    def ahead_outcome_best(self, count: int) -> OutcomePair:
        original_outcome = OutcomePair.outcome_sums(self.ahead(count=count, mode=Mode.PRIORITIZE_ORIGINAL_TEST_RESULTS))
        newest_outcome = OutcomePair.outcome_sums(self.ahead(count=count, mode=Mode.PRIORITIZE_NEWEST_TEST_RESULTS))
        if original_outcome.pass_rate() > newest_outcome.pass_rate():
            return original_outcome
        else:
            return newest_outcome

    def behind(self, count: Optional[int] = None) -> pandas.DataFrame:
        # Note: behind always gets the newest test results for commits behind it.

        # Try to back up <count> from the last time we see this commit in the test records. If this takes us
        # beyond the first time we see the commit, stop.
        starting_index = max(self.index_first_self_test_in_records - count, 0)
        outcomes = self.repo_test_records.iloc[starting_index:min(starting_index+count, self.index_first_self_test_in_records)]
        return outcomes

    @memoized_method(maxsize=10)
    def behind_outcome(self, count: Optional[int] = None) -> OutcomePair:
        return OutcomePair.outcome_sums(self.behind(count=count))

    @memoized_method(maxsize=10)
    def fe(self, window_size: int) -> float:
        return self.ahead_outcome_best(count=window_size).fishers_exact(self.behind_outcome(count=window_size))

    @cached_property
    def fe10(self) -> float:
        return self.fe(10)

    @cached_property
    def mlog10p_10(self):
        return self.ahead_outcome_best(count=10).mlog10Test1t(self.behind_outcome(count=10))

    @cached_property
    def fe10_incoming_slope(self):
        if self.parent:
            return self.fe10 - self.parent.fe10
        return 0.0

    @cached_property
    def fe20(self):
        return self.fe(20)

    @cached_property
    def mlog10p_20(self):
        return self.ahead_outcome_best(count=20).mlog10Test1t(self.behind_outcome(count=20))

    @cached_property
    def fe30(self):
        return self.fe(30)

    @cached_property
    def mlog10p_30(self):
        return self.ahead_outcome_best(count=30).mlog10Test1t(self.behind_outcome(count=30))

    @cached_property
    def worse_than_parent(self) -> bool:
        if not self.parent:
            return False
        return self.parent.fe30 > self.fe30

    @cached_property
    def better_than_parent(self) -> bool:
        if not self.parent:
            return True
        return not self.worse_than_parent

    @cached_property
    def worse_than_child(self) -> bool:
        # if a commit does not have a child, it is the last known commit
        # for a repo. Return True for childless so that it can be considered
        # a valley.
        if not self.child:
            return True
        # Otherwise, return True only if the child fe is better than ours.
        return self.child.fe30 > self.fe30

    @cached_property
    def better_than_child(self) -> bool:
        # if a commit does not have a child, it is the last known commit
        # for a repo. Return True for childless so that it can be considered
        # a peak.
        if not self.child:
            return True
        # Otherwise, return True only if the child fe is worse than ours.
        return not self.worse_than_child

    def is_valley(self):
        return self.worse_than_parent and self.worse_than_child

    def is_peak(self):
        return self.better_than_parent and self.better_than_child

    def github_link(self):
        return f'{self.source_location}/commit/{self.commit_id}'

    @cached_property
    def regression_info(self) -> Tuple[bool, Optional[CommitOutcomes]]:
        """
        :return: Returns True for the first element of the Tuple if this was a regression.
                    The second element of the Tuple is a commit outcomes the system believes resolved
                    the regression. None if the regression is still active.
        """
        if self.fe10 < -0.95 and self.is_valley():
            # Qualifies as possible regression.
            node = self.child
            while node:
                if node.fe10 >= 0:  # If there is a commit causing the pass rate to increase or to have stabilized, see if it resolves us.
                    resolution_fe = node.ahead_outcome_best(count=30).fishers_exact(self.behind_outcome(count=30))
                    if resolution_fe >= -0.5:  # If the signal has improved or is close to statistically insignificant.
                        # We've found an improvement ahead of our valley. Ahead of it
                        # and behind us look very similar statistically. Assume
                        # it resolves us.
                        return True, node
                node = node.child
            return True, None
        else:
            return False, None

    def __str__(self):
        v = f'''Commit: {self.github_link()}
Created: {self.created_at}
Peak: {self.is_peak()}
Valley: {self.is_valley()}
Discrete: {self.discrete_outcomes}
10:
  fe10: {self.mlog10p_10}
20:
  fe20: {self.mlog10p_20}
30:
  fe30: {self.mlog10p_30}
'''
        if self.parent:
            v += f'Parent: {self.parent.commit_id}\n'
        if self.child:
            v += f'Child: {self.child.commit_id}\n'
        return v


SourceLocation = str
CommitId = str
ReleaseName = str


INCLUDE_PR_TESTS = True
LIMIT_ARCH = 'amd64'
LIMIT_NETWORK = '%'  # 'ovn'
LIMIT_PLATFORM = '%'
LIMIT_UPGRADE = '%'  # 'upgrade-micro'

# LIMIT_TEST_ID_SUFFIXES = [list('012345678'), list('90abcdef')]  # Process in two large groups
# LIMIT_TEST_ID_SUFFIXES = list('abcdef0123456789')  # Process in 16 groups
# LIMIT_TEST_ID_SUFFIXES = [f'{r:0>2X}' for r in range(0x100)]  # ids ending with two hex digits; useful for lower memory systems.
# LIMIT_TEST_ID_SUFFIXES = [f'{r:0>3X}' for r in range(0x1000)]  # ids ending with three hex digits; even lower memory
LIMIT_TEST_ID_SUFFIXES = ['5f57267ca12f1857564c93504016c4e3']



def process_queue(input_queue, commits_ordinals):
    for name_group in iter(input_queue.get, 'STOP'):
        name, _ = name_group
        analyze_test_id((name_group, commits_ordinals))
        print(f'Finished {name}')


def analyze_test_id(name_group_commits, grouping_facets=('network', 'upgrade', 'arch', 'platform', 'test_id')):
    name_group, commits_ordinals = name_group_commits
    name, test_id_group = name_group
    grouped_by_facets = test_id_group.groupby(list(grouping_facets), sort=False)
    for name, facets_group in grouped_by_facets:
        # print(f'Processing {name}')

        first_row = facets_group.iloc[0]
        test_name = first_row['test_name']
        network = first_row['network']
        upgrade = first_row['upgrade']
        arch = first_row['arch']
        platform = first_row['platform']
        test_id = first_row['test_id']
        grouping_name = '__'.join(name)

        failures_df = facets_group[(facets_group['success_val'] == 0) & (facets_group['flake_count'] == 0)]
        if len(failures_df.index) < 4:
            # Extremely few failures. Ignore this nurp.
            continue

        # As we visit each test, keep track of the commits we see in the payloads in the order
        # we see them. If we don't have definitive ordering information from github, we will
        # use the order in which commits are observed as a heuristic for parent/child links.
        all_commits: OrderedDict[CommitId, CommitOutcomes] = OrderedDict()

        all_releases: OrderedDict[ReleaseName, ReleasePayload] = OrderedDict()

        for t in facets_group.itertuples():

            source_locations = t.source_locations
            commits = list(t.commits)

            # Associate the commit_ids and the source location for that
            # commit.
            for idx in range(len(commits)):
                commit_id = commits[idx]
                if commit_id not in all_commits:
                    created_at = commits_ordinals.get(commit_id, (None, None))[1]
                    if created_at is None:
                        # If we don't know when the commit was created using github information, then
                        # approximate by the first release to test the commit.
                        release_created = t.release_created
                        created_at = pandas.Timestamp(release_created, tz='UTC')

                    new_commit = CommitOutcomes(source_locations[idx], commit_id, created_at=created_at)
                    all_commits[new_commit.commit_id] = new_commit

                if t.release_name not in all_releases:
                    all_releases[t.release_name] = ReleasePayload(t.release_name, t.release_created)

                all_commits[commit_id].record_observation_in_release(all_releases[t.release_name])

        # There is no guarantee that the order we observe commits being tested is the
        # order in which they merged. Build a definitive order using information from github.
        ordered_commits: List[str] = list(all_commits.keys())

        linked: Set[str] = set()
        linked_commits: Dict[SourceLocation, CommitOutcomes] = dict()
        for commit_to_update in sorted(list(all_commits.values()), key=lambda c: c.created_at):
            commit_id = commit_to_update.commit_id
            source_location = commit_to_update.source_location
            if source_location not in linked_commits:
                # Initial commit found for repo
                linked_commits[source_location] = commit_to_update
                linked.add(commit_id)
            else:
                if commit_id not in linked:
                    linked.add(commit_id)
                    commit_to_update.parent = linked_commits[source_location]
                    linked_commits[source_location].child = commit_to_update
                    linked_commits[source_location] = commit_to_update

        # Make a row for each test result against each commit.
        flattened_nurp_group = facets_group.drop('source_locations', axis=1).explode('commits')  # pandas cannot drop duplicates with a column containing arrays. We don't need source locations in longer, so drop it.
        ordered_commits = list(dict.fromkeys(ordered_commits))  # fast way to remove dupes while preserving order: https://stackoverflow.com/a/17016257
        flattened_nurp_group['commits'] = pandas.Categorical(flattened_nurp_group['commits'], ordered_commits)  # When subsequently sorted, this will sort the entries by their location in the ordered_commits

        # Restore source_locations into newly flattend dataframe. Each commit now has exactly one source location in its row.
        flattened_nurp_group['source_locations'] = flattened_nurp_group.apply(lambda x: all_commits[x.commits].source_location, axis=1)

        # We analyze test results on a repo by repo basis. For efficiency, we groupby
        # source_location. This means sorting, searching, etc, in subsequent methods
        # will only have to deal with relatively small groups of data.
        flattened_groups_by_source_location = flattened_nurp_group.groupby(by='source_locations', sort=False)

        # We now have test results in groups. Each group is all test results for commits in a given
        # upstream repo.
        repo_test_record_groups: Dict[str, pandas.DataFrame] = dict()
        for group_name, group in flattened_groups_by_source_location:
            # Within a release payload, a commit may be encountered multiple times: one
            # for each component it is associated with (e.g. openshift/oc is associated with
            # cli, cli-artifacts, deployer, and tools). We don't want each of these components
            # to count as an individual success/failure against the oc commit, or we will
            # 4x count it. Convert the commits into a set to dedupe.
            # There's a small chance this could multiple successes / flakes / etc. We could fix this
            # by deduping earlier, but in practice the loss should be minimal.
            group.drop_duplicates(inplace=True)

            # ignore_index means the sort will return a dataframe with index=0 for the first item. This is critical
            # for set_data to be able to select appropriate ranges between indices. It's vital to drop duplucates
            # before the sort, or else the index values will not be contiguous.
            group.sort_values(by=['commits', 'modified_time'], ascending=[True, True], inplace=True, ignore_index=True)
            repo_test_record_groups[group_name] = group

        for commit_outcome in all_commits.values():
            commit_outcome.set_data(repo_test_record_groups[commit_outcome.source_location])

        def le_grande_order(commit: CommitOutcomes):
            # Returning a tuple means order by first attribute, then next, then..
            return (
                commit.mlog10p_10,
                commit.mlog10p_20,
                commit.mlog10p_30,
                commit.created_at,  # Resolve disputes with the assumption that the first to merge caused the issue.
            )

        relevant_count = 0
        unresolved_regression: List[CommitOutcomes] = list()
        resolved_regression: List[CommitOutcomes] = list()
        by_le_grande_order: List[CommitOutcomes] = sorted(list(all_commits.values()), key=le_grande_order)

        for c in by_le_grande_order:
            relevant_count += 1
            # print(f'{le_grande_order(c)}')
            regressed, resolution = c.regression_info
            if regressed and not resolution:
                unresolved_regression.append(c)
            if regressed and resolution:
                resolved_regression.append(c)
            # print(c)
            # print()

        # print(f'Found {relevant_count}')
        # print(f'Found {len(unresolved_regression)}')
        # print(f'Found {len(resolved_regression)}')

        analysis_path = pathlib.Path('analysis')
        analysis_path.mkdir(parents=True, exist_ok=True)

        if unresolved_regression:
            output_base = analysis_path.joinpath('unresolved')
        elif resolved_regression:
            output_base = analysis_path.joinpath('resolved')
        else:
            # Don't bother generating a report
            continue

        release_info = facets_group.sort_values(by=['release_created'], ascending=[True])

        unchanged_source_locations: Set[str] = set()

        def fe10_color(fe10):
            if math.isnan(fe10):
                # Indicate visually that there is no semantic value to the
                # record since there is no commit prior to this commit in the
                # test records and thus nothing to compare against.
                return '#888'
            r = 0xff
            g = 0xff
            b = 0xff
            alpha = 0.10  # Only display p-values less than this (i.e. more improbable than this)
            if fe10 < -1 + alpha:
                brightness = (1 - abs(fe10)) / alpha   # the closer fe10 is to -1, the darker we want the red
                r = 100 + int(155 * brightness)
                g = 0
                b = 0
            if fe10 > 1 - alpha:
                brightness = (1 - abs(fe10)) / alpha  # the closer fe10 is to -1, the darker we want the green
                r = 0
                g = 100 + int(155 * brightness)
                b = 0
            return f'#{r:0>2X}{g:0>2X}{b:0>2X}'

        a = Airium()
        a('<!DOCTYPE html>')
        with a.html():
            with a.head():
                a.meta(charset='utf-8')
                a.title(_t=grouping_name)
                with a.style():
                    a('.rb { width: inherit; height: 10px; border: 1px solid #888; box-sizing: border-box;}')
                    a('.rb-unknown { border: 0px solid #888; background-color: #eee;}')  # Used for commits which do not have a commit before them; thus no information to make regression determination.
                    a('.rb-new { border: 2px solid #000; }')  # Highlights newly introduced commits

                    a('th.release-name { height: 140px; white-space: nowrap; }')
                    a('th.release-name > div { transform: translate(0px, 51px) rotate(315deg); width: 15px; }')
                    a('th.release-name > div > span { border-bottom: 1px solid #ccc; }')

                    a('table.results { overflow-y: clip; font-family: monospace; text-align: left; font-size: 8px; line-height: 15px; border-collapse: collapse; border-spacing: 0px; width: 80%; }')
                    a('table.results td { position: relative; padding: 0px; margin: 0px; white-space: nowrap; height: 10px; width: inherit; }')
                    a('table.results tr { padding: 0px; margin: 0px; white-space: nowrap;}')
                    a('table.results th { position: relative; padding: 0px; margin: 0px; white-space: nowrap;}')
                    a('table.results tr:hover { color:blue; background-color: #ffa; }')

                    a('''
a.success:link, a.success:visited {
    color: green;
}                    
a.failure:link, a.failure:visited {
    color: red;
}                    
a.flake:link, a.flake:visited {
    color: gray;
}                    
''')

                    a('''
table.results td:hover::after,
th:hover::after {
  content: "";
  position: absolute;
  background-color: #ffa;
  left: 0;
  top: -5000px;
  height: 10000px;
  width: 100%;
  z-index: -1;
}                    
                    ''')


                    a('''
.styled-table {
    border-collapse: collapse;
    margin: 25px 0;
    font-size: 0.9em;
    font-family: sans-serif;
    min-width: 400px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
}

.styled-table thead tr {
    background-color: #009879;
    color: #ffffff;
    text-align: left;
}

.styled-table th,
.styled-table td {
    padding: 12px 15px;
}

.styled-table tbody tr {
    border-bottom: 1px solid #dddddd;
}

.styled-table tbody tr:nth-of-type(even) {
    background-color: #f3f3f3;
}

.styled-table tbody tr:last-of-type {
    border-bottom: 2px solid #009879;
}

.styled-table tbody tr.active-row {
    font-weight: bold;
    color: #009879;
}
                    ''')
            with a.body():
                with a.h3():
                    a(f'Test: {test_name} ({test_id})')

                if 'upgrade' in grouping_facets:
                    with a.h4():
                        a(f'Upgrade: {upgrade}')

                if 'platform' in grouping_facets:
                    with a.h4():
                        a(f'Platform: {platform}')

                if 'network' in grouping_facets:
                    with a.h4():
                        a(f'Network: {network}')

                if 'arch' in grouping_facets:
                    with a.h4():
                        a(f'Arch: {arch}')

                for release_stream in (ReleasePayloadStreams.CI_PAYLOAD, ReleasePayloadStreams.NIGHTLY_PAYLOAD):
                    z: OrderedDict[str, OrderedDict[str, Tuple]] = OrderedDict()
                    source_locations: OrderedDict[str, bool] = OrderedDict()  # x axis. It is assumed that the source_locations will not change in a given stream over the span of our results

                    for t in release_info.itertuples():
                        if ReleasePayloadStreams.get_stream(t.release_name) is not release_stream:
                            continue

                        if t.release_name in z:
                            # We've already included an analysis of commits relative to this release's release_created date.
                            continue

                        if not source_locations:
                            # If we have not populated source locations for this stream yet, do so.
                            sorted_source_locations = sorted(list(t.source_locations))
                            source_locations.update({sl: True for sl in sorted_source_locations})
                            unchanged_source_locations.update(sorted_source_locations)  # Assume every source location does not have more than one commit until proven later.

                        commit_outcomes: List[CommitOutcomes] = list()
                        for commit_id in t.commits:
                            commit_outcomes.append(all_commits[commit_id])

                        # We need to take care to tolerate if there are new / missing source locations,
                        # but this should be exceedingly rare.
                        commit_outcomes_by_source_location = {commit.source_location: commit for commit in commit_outcomes}

                        z_entry: OrderedDict[str, Tuple] = OrderedDict()
                        for source_location in source_locations.keys():
                            c: CommitOutcomes = commit_outcomes_by_source_location.get(source_location, None)
                            if c:
                                # ahead_outcome = c.ahead_outcome(10, after_date=release_created, mode=Mode.PRIORITIZE_ORIGINAL_TEST_RESULTS)
                                ahead_outcome = c.ahead_outcome_best(10)
                                # fe10 = c.fe10
                                behind_outcome = c.behind_outcome(10)

                                if behind_outcome.failure_count + behind_outcome.success_count > 0:
                                    if source_location in unchanged_source_locations:
                                        # There is at least one commit in this source location that has test runs
                                        # before it was introduced. Make sure to include it in the visualizations /
                                        # analysis.
                                        # This is useful to reduce the amount of information displayed by excluding
                                        # repos that did not change at all during the scan window.
                                        unchanged_source_locations.remove(source_location)
                                    fe10 = ahead_outcome.fishers_exact(behind_outcome)
                                else:
                                    fe10 = math.nan  # If there is nothing behind to compare against, fe is meaningless.

                                # msg = 'ahead:' + str(ahead_outcome) + '&#010;' + 'behind:' + str(behind_outcome) + '&#010;' + 'fe10:' + str(c.fe10) + '&#010;' + source_location + '&#010;' + c.commit_id + '&#010;' + t.release_name
                                msg = 'ahead:' + str(ahead_outcome) + '&#010;' + 'behind:' + str(
                                    behind_outcome) + '&#010;' + 'fe10:' + str(
                                    fe10) + '&#010;' + source_location + '&#010;' + c.commit_id + '&#010;' + t.release_name
                                z_entry[source_location] = (fe10, msg, c)
                            else:
                                z_entry[source_location] = (0.0, '?', None)

                        z[t.release_name] = z_entry

                    if len(z) > 0:
                        release_stream_prefix = ReleasePayloadStreams.split(list(z.keys())[0])[0]  # prefix of first release in results; this should be the same for all other releases in the results.
                        a.br()
                        with a.h4():
                            a(f'Release Stream: {release_stream_prefix}')

                    with a.table(klass="results"):
                        with a.tr():
                            a.th(_t='source')
                            for release_name in z.keys():
                                with a.th(klass='release-name'):
                                    with a.div():
                                        release_name_suffix = ReleasePayloadStreams.split(release_name)[1]
                                        with a.a(href=f'#{release_name}'):
                                            a.span(_t=release_name_suffix)

                        commit_encountered_counts: Dict[str, int] = dict()

                        commits_introduced_during_window = 0
                        for source_location in source_locations.keys():
                            if source_location in unchanged_source_locations:
                                continue
                            with a.tr():
                                repo_name = source_location.split('/')[-1] or '__'  # If no repo, use _ for first and second letter
                                a.td(_t=repo_name)
                                for z_entry in z.values():
                                    fe10, msg, c = z_entry[source_location]

                                    if not c:
                                        # No sha was described in oc adm release info for this component.
                                        # This can happen with CI jobs which formulate payloads based on branch
                                        # names. The test result against this payload are of interest (the
                                        # code that was tested in certainly merged), but we don't know the
                                        # exact commits in the payload at the time.
                                        a.td(title='Payload did not specify commit', _t='?')
                                        continue

                                    commit_encountered_count = commit_encountered_counts.get(c.commit_id, 0)
                                    with a.td(title=msg):
                                        with a.a(href=f'#{c.commit_id}'):
                                            classes = 'rb'
                                            style = ''
                                            if c.parent is None:
                                                classes += ' rb-unknown'
                                            elif commit_encountered_count == 0:
                                                classes += ' rb-new'
                                                style = f'background-color:{fe10_color(fe10)};'
                                                commits_introduced_during_window += 1
                                            elif commit_encountered_count == 1 or fe10 < 0.0:
                                                style = f'background-color:{fe10_color(fe10)};'
                                            a.div(klass=classes, style=style)
                                    commit_encountered_count += 1
                                    commit_encountered_counts[c.commit_id] = commit_encountered_count

                a.h2(_t='Unresolved Regressions')
                if unresolved_regression:
                    for c in unresolved_regression:
                        with a.ul():
                            with a.li():
                                a.a(href=f'#{c.commit_id}', _t=f'{c.repo_name} {c.commit_id}')
                else:
                    a.span(_t='None')

                a.br()

                a.h2(_t='Resolved Regressions')
                if resolved_regression:
                    for c in resolved_regression:
                        with a.ul():
                            with a.li():
                                a.a(href=f'#{c.commit_id}', _t=f'{c.repo_name} {c.commit_id}')
                else:
                    a.span(_t='None')

                a.br()

                a.h2(_t='Commit Details (Highest Possibility of Regression to Lowest)')
                for c in by_le_grande_order:
                    with a.div():
                        with a.a(id=c.commit_id):
                            a.h3(_t=f'Commit: {c.repo_name} {c.commit_id}')

                        with a.table(klass='styled-table'):

                            with a.tr():
                                a.th(_t='Info')
                                a.th(_t='Value')

                            with a.tr():
                                a.td(_t='Link')
                                with a.td():
                                    a.a(href=c.github_link(), _t=c.github_link())

                            with a.tr():
                                a.td(_t='Commit Date')
                                a.td(_t=str(c.created_at))

                            regression, resolution_outcomes = c.regression_info
                            if regression:
                                with a.tr():
                                    a.td(_t='Regression')
                                    a.td(_t=str(regression))

                                with a.tr():
                                    a.td(_t='Regression Resolution')
                                    with a.td():
                                        if resolution_outcomes:
                                            a.a(href=f'#{resolution_outcomes.commit_id}', _t=f'{resolution_outcomes.commit_id}')

                            with a.tr():
                                a.td(_t='Parent')
                                with a.td():
                                    if c.get_parent_commit_id():
                                        a.a(href=f'#{c.get_parent_commit_id()}', _t=c.get_parent_commit_id())

                            with a.tr():
                                a.td(_t='Child')
                                with a.td():
                                    if c.get_child_commit_id():
                                        a.a(href=f'#{c.get_child_commit_id()}', _t=c.get_child_commit_id())

                            if len(c.release_streams[ReleasePayloadStreams.NIGHTLY_PAYLOAD]) > 0:
                                with a.tr():
                                    a.td(_t='First Nightly')
                                    with a.td():
                                        release_name = list(c.release_streams[ReleasePayloadStreams.NIGHTLY_PAYLOAD].keys())[0]
                                        a.a(href=f'#{release_name}', _t=release_name)

                            if len(c.release_streams[ReleasePayloadStreams.CI_PAYLOAD]) > 0:
                                with a.tr():
                                    a.td(_t='First CI Payload')
                                    with a.td():
                                        release_name = list(c.release_streams[ReleasePayloadStreams.CI_PAYLOAD].keys())[0]
                                        a.a(href=f'#{release_name}', _t=release_name)

                        with a.table(klass='styled-table'):
                            with a.tr():
                                a.th(_t='Metric')
                                a.th(_t='Value')
                                a.th(_t='Before Commit')
                                a.th(_t='After Commit')

                            for window_size in (10, 20, 30):
                                with a.tr():
                                    a.td(_t=f'fe{window_size}')
                                    a.td(_t=str(c.fe(window_size)))
                                    a.td(_t=str(c.behind_outcome(window_size)))
                                    a.td(_t=str(c.ahead_outcome_best(window_size)))

                        if c.fe10 >= 0.90 or c.fe10 <= -0.90:
                            with a.table(klass='styled-table'):

                                def render_commit_results(test_runs, only_in_stream: Optional[ReleasePayloadStreams] = None, only_in_prowjob_name: Optional[str] = None):
                                    test_run_count = 0
                                    prowjob_url_successes: OrderedDict[str, int] = OrderedDict()
                                    prowjob_url_failures: OrderedDict[str, int] = OrderedDict()
                                    prowjob_url_flakes: OrderedDict[str, int] = OrderedDict()
                                    for _, row in test_runs.iterrows():
                                        if only_in_stream and ReleasePayloadStreams.get_stream(row['release_name']) is not only_in_stream:
                                            continue
                                        if only_in_prowjob_name and row['prowjob_name'] != only_in_prowjob_name:
                                            continue
                                        prowjob_url = get_prowjob_url(row)
                                        prowjob_url_successes[prowjob_url] = prowjob_url_successes.get(prowjob_url, 0) + row['success_val']
                                        prowjob_url_flakes[prowjob_url] = prowjob_url_flakes.get(prowjob_url, 0) + row['flake_count']
                                        prowjob_url_failures[prowjob_url] = prowjob_url_failures.get(prowjob_url, 0) + (1 - row['success_val']) - row['flake_count']

                                    for prowjob_url in prowjob_url_successes.keys():
                                        for _ in range(prowjob_url_successes[prowjob_url]):
                                            a.a(href=prowjob_url, klass='testr success', _t='S', target="_blank")
                                        for _ in range(prowjob_url_flakes[prowjob_url]):
                                            a.a(href=prowjob_url, klass='testr flake', _t='f', target="_blank")
                                        for _ in range(max(prowjob_url_failures[prowjob_url], 0)):
                                            a.a(href=prowjob_url, klass='testr failure', _t='F', target="_blank")
                                        test_run_count += 1
                                        if test_run_count % 20 == 0:
                                            a.br()

                                with a.tr():
                                    a.th(_t='Test Type')
                                    a.th(_t='Before Commit (Up to 30)')
                                    a.th(_t='After Commit (Up to 30)')

                                behind_test_results = c.behind(30)
                                ahead_test_results = c.ahead(30)

                                prowjob_names: Set[str] = set()
                                for _, row in behind_test_results.iterrows():
                                    prowjob_names.add(row['prowjob_name'])

                                for _, row in ahead_test_results.iterrows():
                                    prowjob_names.add(row['prowjob_name'])

                                with a.tr():
                                    a.td(_t='*')
                                    with a.td(style='font-family: monospace; text-align: left; vertical-align: top;'):
                                        render_commit_results(behind_test_results)
                                    with a.td(style='font-family: monospace; text-align: left;  vertical-align: top;'):
                                        render_commit_results(ahead_test_results)

                                for stream in ReleasePayloadStreams:
                                    with a.tr():
                                        a.td(_t=f'{stream.value}')
                                        with a.td(style='font-family: monospace; text-align: left;'):
                                            render_commit_results(behind_test_results, only_in_stream=stream)

                                        with a.td(style='font-family: monospace; text-align: left;'):
                                            render_commit_results(ahead_test_results, only_in_stream=stream)

                                with a.tr():
                                    a.th(_t='Prowjobs')
                                    a.th(_t='Before Commit (Up to 30)')
                                    a.th(_t='After Commit (Up to 30)')

                                for prowjob_name in prowjob_names:
                                    with a.tr():
                                        a.td(_t=f'{prowjob_name}')
                                        with a.td(style='font-family: monospace; text-align: left;'):
                                            render_commit_results(behind_test_results, only_in_prowjob_name=prowjob_name)

                                        with a.td(style='font-family: monospace; text-align: left;'):
                                            render_commit_results(ahead_test_results, only_in_prowjob_name=prowjob_name)

                        a.br()
                        a.br()

                a.h2('Release Details')
                for release_name, release_payload in all_releases.items():
                    with a.div():
                        with a.a(id=release_name):
                            a.h3(_t=f'Release: {release_name}')
                        a.h4(_t='Commits Introduced')
                        with a.table(klass='styled-table'):
                            with a.tr():
                                a.th(_t='Repo')
                                a.th(_t='Commit')
                                a.th(_t='Value (10)')
                                a.th(_t='Before Commit (10)')
                                a.th(_t='After Commit (10)')
                            for commit_id in sorted(release_payload.diff_commits, key=lambda cid: all_commits[cid].fe10):
                                with a.tr():
                                    c = all_commits[commit_id]
                                    a.td(_t=c.repo_name)
                                    with a.td():
                                        a.a(href=f'#{c.commit_id}', _t=c.commit_id)
                                    a.td(_t=str(c.fe10))
                                    a.td(_t=str(c.behind_outcome(10)))
                                    a.td(_t=str(c.ahead_outcome_best(10)))

        output_base.mkdir(parents=True, exist_ok=True)
        facets_path = output_base.joinpath('by_' + '_'.join(grouping_facets))
        facets_path.mkdir(parents=True, exist_ok=True)
        output_path = facets_path.joinpath(f'{grouping_name}.html')

        with output_path.open(mode='w+') as f:
            f.write(str(a))

        # There was a regression for this test_id, so if we are grouping by all facets
        # in this invocation, analyze the test again, but only using two facets. Also
        # run just for the test_id in isolation in case it is not influenced by
        # environment or configuration.
        if len(grouping_facets) > 2:
            analyze_test_id(name_group_commits, grouping_facets=('network', 'test_id'))
            analyze_test_id(name_group_commits, grouping_facets=('upgrade', 'test_id'))
            analyze_test_id(name_group_commits, grouping_facets=('platform', 'test_id'))
            analyze_test_id(name_group_commits, grouping_facets=('arch', 'test_id'))
            analyze_test_id(name_group_commits, grouping_facets=('test_id',))


@click.command()
@click.option('-r', '--release', required=True, help='OpenShift release to analyze (e.g. "4.15")')
def cli(release):
    scan_period_days = 14  # days
    before_datetime = datetime.datetime.utcnow()
    # before_datetime = datetime.datetime(2023, 8, 1, 0, 0)  # There was a regression for GCP and Azure on 7/20
    # before_datetime = datetime.datetime(2023, 7, 24, 4, 10, 0)  # This is a time shortly before the first test results including the revert to the Azure regression.
    after_datetime = before_datetime - datetime.timedelta(days=scan_period_days)
    before_str = before_datetime.strftime("%Y-%m-%d %H:%M:%S")
    after_str = after_datetime.strftime("%Y-%m-%d %H:%M:%S")
    scan_period = f'BETWEEN DATETIME "{after_str}" AND DATETIME "{before_str}"'

    main_client = bigquery.Client(project='openshift-gce-devel')

    find_commits = ''
    date_stepper = before_datetime

    for datestep in (1, 2):
        try:
            for i in range(scan_period_days+1):

                if find_commits:
                    find_commits += '\nUNION ALL\n'

                find_commits += f'''
                SELECT created_at, CONCAT("github.com/", repo.name) as repo, 
                    ARRAY(
                        SELECT JSON_EXTRACT_SCALAR(x, '$.sha')
                        FROM UNNEST(JSON_EXTRACT_ARRAY(payload,'$.commits' )) x
                    ) as commits
                FROM `githubarchive.day.{date_stepper.strftime("%Y%m%d")}` 
                WHERE
                    type = "PushEvent"
                    AND (repo.name LIKE "operator-framework/%" OR repo.name LIKE "openshift/%") 
                '''
                date_stepper = date_stepper - datetime.timedelta(days=1)

            find_commits = f'''
                WITH commits AS (
                    {find_commits}
                )   SELECT *
                    FROM commits
                    ORDER BY created_at ASC
            '''

            print('Gathering github commit information...')

            # Each row includes an array of commit shas.
            commits_bulk = main_client.query(find_commits).to_dataframe(create_bqstorage_client=True, progress_bar_type='tqdm')

            commits_info = pandas.DataFrame(
                columns=[
                    'created_at',
                    'repo',
                    'commit',
                ]
            )

            commits_info_idx = 0
            for row in commits_bulk.itertuples():
                for commit in row.commits:
                    commits_info.loc[commits_info_idx] = [
                        row.created_at,
                        row.repo,
                        commit,
                    ]
                    commits_info_idx += 1

            # Commits info should now contain all commits merged into the openshift repos.
            # This will include merge commits as well as (up to 20 -- due to PushEvent payload
            # limitations) commits they carried into the repo.
            commits_info.drop_duplicates()
            # Success!
            break
        except:
            # Current day may not be in archive, yet, back up a single day.
            find_commits = ''
            date_stepper = before_datetime - datetime.timedelta(days=1)
            if datestep == 2:
                # Missing multiple days in github archive?
                raise

    query_commits_tested_by_non_pr_payloads = f'''
    SELECT DISTINCT(tag_commit_id) as commit FROM `openshift-gce-devel.ci_analysis_us.job_releases` 
        WHERE is_pr = false
        AND release_name LIKE "{release}.%"
        AND release_created {scan_period}
    '''

    print('Finding commits tested by non-pr release payloads')
    commits_tested_by_non_pr_payloads_df = main_client.query(query_commits_tested_by_non_pr_payloads).to_dataframe(create_bqstorage_client=True, progress_bar_type='tqdm')
    commits_tested_by_non_pr_payloads = commits_tested_by_non_pr_payloads_df['commit'].tolist()
    print(f'Found {len(commits_tested_by_non_pr_payloads)} of them')

    # Construct a definitive set including all commits that ultimately merged during
    # our observation period.
    merged_commits = set(commits_info['commit'].tolist())
    merged_commits.update(commits_tested_by_non_pr_payloads)

    commits_ordinals: Dict[str, Tuple[int, datetime.datetime]] = dict()
    # Create a dict mapping each commit to an increasing integer. This will allow more
    # efficient lookup later when we need to know whether one commit came after another.
    count = 0
    for row in commits_info.itertuples():
        commits_ordinals[row.commit] = (count, row.created_at)
        count += 1

    queue = multiprocessing.Queue(os.cpu_count() * 300)
    worker_pool = [multiprocessing.Process(target=process_queue, args=(queue, commits_ordinals)) for _ in range(max(os.cpu_count() - 2, 1))]
    for worker in worker_pool:
        worker.start()

    for test_id_suffix in LIMIT_TEST_ID_SUFFIXES:

        if test_id_suffix == '*':
            suffix_test = ''
        elif type(test_id_suffix) != list:
            # If it is not a list, turn it into one
            test_id_suffix = [test_id_suffix]

        # If an entry is like ['a', 'b', 'c'], then we want to convert that into a logical
        # or 'ends_with(a) or ends_with(b) or ends_with(c)'.
        if type(test_id_suffix) == list:
            suffix_test = ''
            for suffix in test_id_suffix:
                suffix_test += f'OR ENDS_WITH(test_id, "{suffix}") '
            suffix_test = suffix_test[3:]  # Strip initial OR

        if suffix_test:
            suffix_test = f'AND ({suffix_test})'

        suffixed_records = f'''
            WITH junit_all AS(
                
                # Find 4.x prowjobs which tested payloads during the scan period. For each
                # payload, aggregate the commits it included into an array.
                WITH payload_components AS(               
                    # Find all prowjobs which have run against a 4.x payload commit
                    # in the last two months. 
                    SELECT  prowjob_build_id as pjbi, 
                            ARRAY_AGG(tag_source_location) as source_locations, 
                            ARRAY_AGG(tag_commit_id) as commits, 
                            ANY_VALUE(release_name) as release_name, 
                            ANY_VALUE(release_created) as release_created 
                    FROM openshift-gce-devel.ci_analysis_us.job_releases jr
                    WHERE   release_created {scan_period}
                            AND (release_name LIKE "{release}.0-0.nightly%" OR release_name LIKE "{release}.0-0.ci%")   
                    GROUP BY prowjob_build_id
                )
    
                # Find all junit tests run in non-PR triggered tests which ran as part of those prowjobs over the last two months
                SELECT  modified_time,
                        test_id,
                        prowjob_name,
                        junit.prowjob_build_id AS prowjob_build_id,
                        network,
                        upgrade,
                        arch,
                        platform,
                        success_val,
                        flake_count,
                        test_name,
                        commits,
                        source_locations,
                        release_name,
                        release_created,
                        0 AS is_pr, 
                        "N/A" as pr_sha,
                        jobs.prowjob_url as prowjob_url
                FROM    `openshift-gce-devel.ci_analysis_us.junit` junit 
                        INNER JOIN payload_components ON junit.prowjob_build_id = payload_components.pjbi 
                        INNER JOIN `openshift-gce-devel.ci_analysis_us.jobs` jobs ON junit.prowjob_build_id = jobs.prowjob_build_id 
                WHERE   arch LIKE "{LIMIT_ARCH}"
                        AND network LIKE "{LIMIT_NETWORK}"
                        AND junit.modified_time {scan_period}
                        {suffix_test}
                        # AND test_name NOT LIKE "%disruption%"
                        AND platform LIKE "{LIMIT_PLATFORM}" 
                        AND upgrade LIKE "{LIMIT_UPGRADE}" 
                UNION ALL    
    
                # Find all junit tests run in PR triggered tests which ran as part of those prowjobs over the last two months
                SELECT  modified_time,
                        test_id,
                        prowjob_name,
                        junit_pr.prowjob_build_id AS prowjob_build_id,
                        network,
                        upgrade,
                        arch,
                        platform,
                        success_val,
                        flake_count,
                        test_name,
                        commits,
                        source_locations,
                        release_name,
                        release_created,
                        1 AS is_pr,
                        jobs.pr_sha AS pr_sha,
                        jobs.prowjob_url as prowjob_url
                FROM    `openshift-gce-devel.ci_analysis_us.junit_pr` junit_pr 
                        INNER JOIN payload_components ON junit_pr.prowjob_build_id = payload_components.pjbi
                        INNER JOIN `openshift-gce-devel.ci_analysis_us.jobs` jobs ON junit_pr.prowjob_build_id = jobs.prowjob_build_id 
                WHERE   arch LIKE "{LIMIT_ARCH}"
                        AND network LIKE "{LIMIT_NETWORK}"
                        AND junit_pr.modified_time {scan_period}
                        {suffix_test} 
                        # AND test_name NOT LIKE "%disruption%" 
                        AND platform LIKE "{LIMIT_PLATFORM}" 
                        AND upgrade LIKE "{LIMIT_UPGRADE}"
                        AND jobs.repo != "release"  # Ignore rehearsals 
            )
    
            SELECT  *
            FROM junit_all
            ORDER BY modified_time ASC
    '''

        print(f'Gathering test runs for suffix: {test_id_suffix}')
        all_records = main_client.query(suffixed_records).to_dataframe(create_bqstorage_client=True, progress_bar_type='tqdm')

        all_record_count = len(all_records.index)
        pr_record_count = len(all_records[all_records['is_pr'] == 1])
        print(f'Non-PR Records {all_record_count-pr_record_count}')
        print(f'    PR Records {pr_record_count}')

        if INCLUDE_PR_TESTS:
            # If a pr triggered test tested commits that are now a subset of all trusted / merged
            # commits, then promote the pr test result to a non-pr test result.
            all_records.loc[
                ((all_records['is_pr'] == 1) & (all_records['pr_sha'].isin(merged_commits))), 'is_pr'
            ] = 0

        # Drop all records that were pre-merge tests containing untrusted commits.
        trusted_records = all_records[all_records['is_pr'] == 0]
        trusted_record_count = len(trusted_records.index)
        print(f'Dropped {all_record_count-trusted_record_count} untrusted records')

        print(f'There are {len(trusted_records)} records to process with suffix: {test_id_suffix}')
        grouped_by_test_id = trusted_records.groupby('test_id', sort=False)
        print(f'{len(grouped_by_test_id.groups)} different test_ids in this suffix')
        for input_item in grouped_by_test_id:
            queue.put(input_item)

    for worker in worker_pool:
        queue.put('STOP')

    for worker in worker_pool:
        worker.join()


if __name__ == '__main__':
    cli()
