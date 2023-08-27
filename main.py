#!/usr/bin/env python3
import math
import multiprocessing
import datetime
import traceback
from typing import NamedTuple, List, Dict, Set, Tuple, Optional
from enum import Enum
import time
import re

import numpy
import tqdm
from collections import OrderedDict

from airium import Airium

import itertools
import pandas
from google.cloud import bigquery
from fast_fisher import fast_fisher_cython
from functools import cached_property, lru_cache

import plotly.graph_objects as go
import plotly.express as px

pandas.options.compute.use_numexpr = True


class Mode(Enum):
    PRIORITIZE_NEWEST_TEST_RESULTS = 0  # Useful if we are triggering new tests to evaluate signal or want to see if a signal is falling
    PRIORITIZE_ORIGINAL_TEST_RESULTS = 1


RELEASE_NAME_SPLIT_REGEX = re.compile(r"(.*-0\.[^-]+)-(.*)")


class PayloadStreams(Enum):
    CI_PAYLOAD = 'ci'
    NIGHTLY_PAYLOAD = 'nightly'
    PR_PAYLOAD = 'pr'

    @staticmethod
    def get_stream(release_name: str):
        if '-0.ci-' in release_name:
            return PayloadStreams.CI_PAYLOAD

        if '-0.nightly-' in release_name:
            return PayloadStreams.NIGHTLY_PAYLOAD

        if '-0.ci.test-' in release_name:
            return PayloadStreams.PR_PAYLOAD

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

    def __init__(self, commit_id, success_count: int, failure_count: int, flake_count: int):
        self.success_count = success_count
        self.failure_count = failure_count
        self.flake_count = flake_count
        self.commit_id = commit_id

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

    def __str__(self):
        return f'[s={self.success_count} f={self.failure_count} r={int(self.pass_rate() * 100)}%]'


MAX_WINDOW_SIZE = 30


class CommitOutcomes:

    def __init__(self, source_location: str, commit_id: str, created_at: Optional[pandas.Timestamp]):
        self.source_location = source_location
        self.commit_id = commit_id
        self.created_at = created_at
        self.parent = None
        self.child = None

        self.repo_test_records: Optional[pandas.DataFrame] = None
        self.index_first_self_test_in_records = 0
        self.index_last_self_test_in_records = 0

        self.discrete_outcomes: Optional[OutcomePair] = None
        self.tests_against_this_commit: Optional[pandas.DataFrame] = None

        self.release_streams: Dict[PayloadStreams, OrderedDict[str, bool]] = {
            PayloadStreams.NIGHTLY_PAYLOAD: OrderedDict(),  # We really just want an Ordered Set. Value doesn't matter.
            PayloadStreams.CI_PAYLOAD: OrderedDict(),
            PayloadStreams.PR_PAYLOAD: OrderedDict()
        }

    def set_data(self, repo_test_records: pandas.DataFrame):
        self.repo_test_records = repo_test_records

        self.tests_against_this_commit = self.repo_test_records[self.repo_test_records['commits'] == self.commit_id]
        self.discrete_outcomes = self.outcome_sums(self.tests_against_this_commit)

        self.index_first_self_test_in_records = self.tests_against_this_commit.first_valid_index()
        self.index_last_self_test_in_records = self.tests_against_this_commit.last_valid_index()

    def record_observation_in_release(self, release_name: str):
        stream_name = PayloadStreams.get_stream(release_name)
        if stream_name:
            self.release_streams[stream_name][release_name] = True

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

    def outcome_sums(self, selection: pandas.DataFrame) -> OutcomePair:
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

        return OutcomePair(self.commit_id, success_count=success_count, failure_count=failure_count, flake_count=flake_count)

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
        return self.outcome_sums(self.ahead(count=count, mode=mode, after_date=after_date))

    @lru_cache(maxsize=10)
    def ahead_outcome_best(self, count: int) -> OutcomePair:
        original_outcome = self.outcome_sums(self.ahead(count=count, mode=Mode.PRIORITIZE_ORIGINAL_TEST_RESULTS))
        newest_outcome = self.outcome_sums(self.ahead(count=count, mode=Mode.PRIORITIZE_NEWEST_TEST_RESULTS))
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

    @lru_cache(maxsize=10)
    def behind_outcome(self, count: Optional[int] = None) -> OutcomePair:
        return self.outcome_sums(self.behind(count=count))

    @cached_property
    def fe10(self) -> float:
        return self.ahead_outcome_best(count=10).fishers_exact(self.behind_outcome(count=10))

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
        return self.ahead_outcome_best(count=20).fishers_exact(self.behind_outcome(count=20))

    @cached_property
    def mlog10p_20(self):
        return self.ahead_outcome_best(count=20).mlog10Test1t(self.behind_outcome(count=20))

    @cached_property
    def fe30(self):
        return self.ahead_outcome_best(count=30).fishers_exact(self.behind_outcome(count=30))

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

    def _behind_scan(self) -> Tuple[int, int]:
        successes = self.discrete_outcomes.success_count
        failures = self.discrete_outcomes.failure_count

        if self.worse_than_parent:
            node = self
            while node.parent and node.worse_than_parent:  # traverse as a valley
                successes += node.parent.discrete_outcomes.success_count
                failures += node.parent.discrete_outcomes.failure_count
                node = node.parent

        else:
            node = self
            while node.parent and node.better_than_parent:
                successes += node.parent.discrete_outcomes.success_count
                failures += node.parent.discrete_outcomes.failure_count
                node = node.parent

        return successes, failures

    @cached_property
    def behind_dynamic(self) -> OutcomePair:
        successes, failures = self._behind_scan()
        behind = OutcomePair(-1, self.commit_id)
        # Looking behind us, we should not include our own successes/failures
        behind.add_success(amount=successes - self.discrete_outcomes.success_count)
        behind.add_failure(amount=failures - self.discrete_outcomes.failure_count)
        return behind

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

    def _ahead_scan(self) -> Tuple[int, int]:
        successes = self.discrete_outcomes.success_count
        failures = self.discrete_outcomes.failure_count

        if self.worse_than_child:
            node = self
            while node.child and node.worse_than_child and node.child.fe10 <= 0:  # traverse as a valley. Stop if child is improving.
                successes += node.child.discrete_outcomes.success_count
                failures += node.child.discrete_outcomes.failure_count
                node = node.child

        else:
            node = self
            while node.child and node.better_than_child and node.child.fe10 >= 0:  # traverse as peak. Stop if child is regressing.
                successes += node.child.discrete_outcomes.success_count
                failures += node.child.discrete_outcomes.failure_count
                node = node.child

        return successes, failures

    @cached_property
    def ahead_dynamic(self) -> OutcomePair:
        successes, failures = self._ahead_scan()
        ahead = OutcomePair(-1, commit_id=self.commit_id)
        ahead.add_success(amount=successes)
        ahead.add_failure(amount=failures)
        return ahead

    def is_valley(self):
        return self.worse_than_parent and self.worse_than_child

    def is_peak(self):
        return self.better_than_parent and self.better_than_child

    def github_link(self):
        return f'{self.source_location}/commit/{self.commit_id}'

    def regression_info(self):
        """
        :return: Returns True for the first element of the Tuple if this was a regression.
                    The second element of the Tuple is a commit outcomes the system believes resolved
                    the regression. None if the regression is still active.
        """
        if self.fe10 < -0.95 and self.is_valley():
            # Qualifies as regression.
            node = self.child
            while node:
                if node.fe10 >= 0:  # If there is a commit causing the pass rate to increase or to have stabilized, see if it resolves us.
                    resolution_fe = node.ahead_outcome(count=30).fishers_exact(self.behind_outcome(count=30))
                    if resolution_fe >= -0.5:  # If the signal has improved or is close to statistically insignificant.
                        # We've found a peak in front of our valley. Ahead of it
                        # and behind us look very similar statistically. Assume
                        # it resolves us.
                        return True, node.github_link()
                node = node.child
            return True, None
        else:
            return False, None

    @cached_property
    def fe_dynamic(self):
        behind = self.behind_dynamic
        ahead = self.ahead_dynamic
        return ahead.fishers_exact(behind)

    def __str__(self):
        v = f'''Commit: {self.github_link()}
Created: {self.created_at}
Peak: {self.is_peak()}
Valley: {self.is_valley()}
Discrete: {self.discrete_outcomes}
Regression Info: {self.regression_info()}
10:
  fe10: {self.mlog10p_10}
20:
  fe20: {self.mlog10p_20}
30:
  fe30: {self.mlog10p_30}
dynamic:
  behind: {{self.behind_dynamic}}
  ahead: {{self.ahead_dynamic}}
  fe: {{self.fe_dynamic}}
'''
        if self.parent:
            v += f'Parent: {self.parent.commit_id}\n'
        if self.child:
            v += f'Child: {self.child.commit_id}\n'
        return v


SourceLocation = str
CommitId = str


INCLUDE_PR_TESTS = True
LIMIT_ARCH = 'amd64'
LIMIT_NETWORK = '%'  # 'ovn'
LIMIT_PLATFORM = 'gcp'
LIMIT_UPGRADE = '%'  # 'upgrade-micro'
# LIMIT_TEST_ID_SUFFIXES = list('abcdef0123456789')  # ids end with a hex digit, so cover everything.
# LIMIT_TEST_ID_SUFFIXES = list('56789')  # ids end with a hex digit, so cover everything.
LIMIT_TEST_ID_SUFFIXES = ['9d46f2845cf09db01147b356db9bfe0d']


def analyze_test_id(name_group_commits):
    name_group, commits_ordinals = name_group_commits
    name, test_id_group = name_group
    # grouped_by_nurp = test_id_group.groupby(['network', 'upgrade', 'arch', 'platform', 'test_id'], sort=False)
    grouped_by_nurp = test_id_group.groupby(['platform', 'test_id'], sort=False)
    for name, nurp_group in grouped_by_nurp:
        # print(f'Processing {name}')

        # As we visit each test, keep track of the commits we see in the payloads in the order
        # we see them. If we don't have definitive ordering information from github, we will
        # use the order in which commits are observed as a heuristic for parent/child links.
        all_commits: OrderedDict[CommitId, CommitOutcomes] = OrderedDict()

        first_row = nurp_group.iloc[0]
        test_name = first_row['test_name']
        network = first_row['network']
        upgrade = first_row['upgrade']
        arch = first_row['arch']
        platform = first_row['platform']
        test_id = first_row['test_id']
        qtest_id = f"{network}_{upgrade}_{arch}_{platform}_{test_id}"

        for t in nurp_group.itertuples():
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
                all_commits[commit_id].record_observation_in_release(t.release_name)

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
        flattened_nurp_group = nurp_group.drop('source_locations', axis=1).explode('commits')  # pandas cannot drop duplicates with a column containing arrays. We don't need source locations in longer, so drop it.
        ordered_commits = list(dict.fromkeys(ordered_commits))  # fast way to remove dupes while preserving order: https://stackoverflow.com/a/17016257
        flattened_nurp_group['commits'] = pandas.Categorical(flattened_nurp_group['commits'], ordered_commits)  # When subsequently sorted, this will sort the entries by their location in the ordered_commits

        # Restore source_locations into newly flattend dataframe. Each commit now has exactly one source location in its row.
        flattened_nurp_group['source_locations'] = flattened_nurp_group.apply(lambda x: all_commits[x.commits].source_location, axis=1)

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
        # Assuming that we iterate forward through these results to populate the commit
        # aggregators (i.e. ahead and behind successes/failures), this ordering is not
        # ideal.
        #
        # 1. When informing self about successes/failures ahead, we want to prioritize the
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

        # We analyze test results on a repo by repo basis. For efficiency, we groupby
        # source_location. This means sorting, searching, etc, in subsequent methods
        # will only have to deal with relatively small groups of data.
        flattened_groups_by_source_location = flattened_nurp_group.groupby(by='source_locations', sort=False)

        # We are about to process test results for self+ancestors. See long comment
        # above about why commits are ascending order (order of commit history within
        # a given repo) and modified_time (test run time) is descending (reverse
        # chronological).
        repo_test_record_groups: Dict[str, pandas.DataFrame] = dict()
        for group_name, group in flattened_groups_by_source_location:
            # Within a release payload, a commit may be encountered multiple times: one
            # for each component it is associated with (e.g. openshift/oc is associated with
            # cli, cli-artifacts, deployer, and tools). We don't want each these components
            # count an individual success/failure against the oc commit, or we will
            # 4x count it. Convert the commits into a set to dedupe.
            # There's a small chance this could multiple successes / flakes / etc. We could fix this
            # by deduping earlier, but in practice the loss should be minimal.
            group.drop_duplicates(inplace=True)

            # ignore_index means the sort will return a dataframe with index=0 for the first item. This is critical
            # for set_data to be able to select appropriate ranges between indices. It's vital to drop duplucates
            # before the sort, or else the index values will not be contiguous.
            group.sort_values(by=['commits', 'modified_time'], ascending=[True, True], inplace=True, ignore_index=True)
            repo_test_record_groups[group_name] = group

        # import cProfile, pstats, io
        # from pstats import SortKey
        # pr = cProfile.Profile()
        # pr.enable()

        for commit_outcome in all_commits.values():
            commit_outcome.set_data(repo_test_record_groups[commit_outcome.source_location])

        def le_grande_order(commit: CommitOutcomes):
            # Returning a tuple means order by first attribute, then next, then..
            return (
                #commit.fe_dynamic,
                #commit.fe10_incoming_slope,
                #commit.discrete_outcomes.pass_rate(),
                #-1 * commit.ahead_dynamic.failure_count,
                commit.mlog10p_10,
                commit.mlog10p_20,
                commit.mlog10p_30,
                commit.created_at,  # Resolve disputes with the assumption that the first to merge caused the issue.
            )

        relevant_count = 0
        unresolved_regression: List[str] = list()
        resolved_regression: List[str] = list()
        by_le_grande_order: List[CommitOutcomes] = sorted(list(all_commits.values()), key=le_grande_order)

        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

        for c in by_le_grande_order:
            # if c.fe_dynamic > -0.98:
            #     break
            # if not c.is_valley():
            #     continue
            relevant_count += 1
            print(f'{le_grande_order(c)}')
            regressed, resolution = c.regression_info()
            if regressed and not resolution:
                unresolved_regression.append(c.commit_id)
            if regressed and resolution:
                resolved_regression.append(c.commit_id)
            print(c)
            print()

        print(f'Found {relevant_count}')
        print(f'Found {len(unresolved_regression)} unresolved regressions: {unresolved_regression}')
        print(f'Found {len(resolved_regression)} resolved regressions: {resolved_regression}')

        commit_ids = []
        z: OrderedDict[str, OrderedDict[str, Tuple]] = OrderedDict()
        release_names: OrderedDict[str, bool] = OrderedDict()  # y axis
        source_locations: List = None  # x axis. It is assumed that the source_locations will not change in a given stream over the span of our results

        release_info = nurp_group.sort_values(by=['release_created'], ascending=[True])

        unchanged_source_locations: Set[str] = set()

        for t in release_info.itertuples():
            if PayloadStreams.get_stream(t.release_name) != PayloadStreams.NIGHTLY_PAYLOAD:
                continue

            if t.release_name in z:
                # We've already included an analysis of commits relative to this release's release_created date.
                continue

            if not source_locations:
                source_locations = sorted(list(t.source_locations))
                unchanged_source_locations.update(source_locations)  # Assume every source location does not have more than one commit until proven later.

            release_created = t.release_created

            commit_outcomes: List[CommitOutcomes] = list()
            for commit_id in t.commits:
                commit_outcomes.append(all_commits[commit_id])

            # We need to take care to tolerate if there are new / missing source locations,
            # but this should be exceedingly rare.
            commit_outcomes_by_source_location = {commit.source_location: commit for commit in commit_outcomes}

            z_entry: OrderedDict[str, Tuple] = OrderedDict()
            commits_entry = []
            for source_location in source_locations:
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
            commit_ids.append(commits_entry)

        def fe10_color(fe10):
            if math.isnan(fe10):
                # Indicate visually that there is no semantic value to the
                # record since there is no commit prior to this commit in the
                # test records and thus nothing to compare against.
                return '#888'
            r = 0xff
            g = 0xff
            b = 0xff
            if fe10 < -0.90:
                r = 100 + int(155 * (1 - abs(fe10)))
                g = 0
                b = 0
            if fe10 > 0.90:
                r = 0
                g = 100 + int(155 * (1 - abs(fe10)))
                b = 0
            return f'#{r:0>2X}{g:0>2X}{b:0>2X}'

        repos_by_first_letter: OrderedDict[str, OrderedDict[str, bool]] = OrderedDict()
        for source_location in source_locations:
            if source_location in unchanged_source_locations:
                continue

            repo_name = source_location.split('/')[-1] or '__'  # If no repo, use _ for first and second letter
            first_letter = repo_name[:1]
            if first_letter not in repos_by_first_letter:
                repos_by_first_letter[first_letter] = OrderedDict()
            repos_by_first_letter[first_letter][repo_name] = True

        a = Airium()
        a('<!DOCTYPE html>')
        with a.html():
            with a.head():
                a.meta(charset='utf-8')
                a.title(_t=qtest_id)
                with a.style():
                    a('.rb { width: 8px; height: 10px; border: 1px solid #888; box-sizing: border-box;}')
                    a('.rb-unknown { width: 8px; height: 10px; border: 0px solid #888; background-color: #eee; box-sizing: border-box;}')
                    a('.rb-new { width: 8px; height: 10px; border: 1px solid #00d; box-sizing: border-box; }')
                    a('table.results { font-family: monospace; text-align: left; font-size: 8px; line-height: 10px; border-collapse: collapse; border-spacing: 0px; }')
                    a('table.results td { padding: 0px; margin: 0px; white-space: nowrap; height: 10px; width: 11px; }')
                    a('table.results tr { padding: 0px; margin: 0px; white-space: nowrap;}')
                    a('table.results th { padding: 0px; margin: 0px; white-space: nowrap;}')
            with a.body():
                with a.h3():
                    a(f'Test: {test_name} ({test_id}')

                with a.h4():
                    a(f'Upgrade: {upgrade}')

                with a.h4():
                    a(f'Platform: {platform}')

                with a.h4():
                    a(f'Network: {network}')

                with a.h4():
                    a(f'Arch: {arch}')

                if len(z) > 0:
                    release_stream_prefix = PayloadStreams.split(list(z.keys())[0])[0]  # prefix of first release in results; this should be the same for all other releases in the results.
                    with a.h5():
                        a(f'Release Stream: {release_stream_prefix}')

                with a.table(klass="results"):
                    with a.tr():
                        a.th(_t='release')
                        for first_letter, sl_list in repos_by_first_letter.items():
                            a.th(_t=first_letter, colspan=f'{len(sl_list)}')

                    with a.tr():
                        a.th(_t='')  # release column
                        for _, sl_list in repos_by_first_letter.items():
                            for repo_name in sl_list:
                                a.th(_t=repo_name[1:2], title=repo_name)  # second letter

                    commit_encountered_counts: Dict[str, int] = dict()
                    for release_name, z_entry in z.items():
                        with a.tr():
                            a.td(_t=PayloadStreams.split(release_name)[1])  # release name suffix
                            for source_location, item_tuple in z_entry.items():
                                fe10, msg, c = item_tuple
                                if source_location in unchanged_source_locations:
                                    continue
                                commit_encountered_count = commit_encountered_counts.get(c.commit_id, 0)
                                with a.td(title=msg):
                                    if c.parent is None:
                                        a.div(klass='rb-unknown')
                                    elif commit_encountered_count == 0:
                                        a.div(klass='rb-new', style=f'background-color:{fe10_color(fe10)};')
                                    elif commit_encountered_count == 1 or fe10 < 0.0:
                                        a.div(klass='rb', style=f'background-color:{fe10_color(fe10)};')
                                    else:
                                        a.div(klass='rb')
                                commit_encountered_count += 1
                                commit_encountered_counts[c.commit_id] = commit_encountered_count

        with open('airium.html', mode='w+') as f:
            f.write(str(a))

        try:
            # fig = px.imshow(z)
            fig = go.Figure(data=go.Heatmap(
                z=df_z.values,
                text=df_commit_ids,
                # x=source_locations,
                # y=list(release_names.keys())
            ))
            fig.update_layout(
                yaxis=dict(
                    visible=False,
                ),
                xaxis=dict(
                    visible=False,
                )
            )
            fig.show()
            pass
            # fig = go.Figure(data=[go.Surface(z=z)])
            # fig.update_layout(title='Mt Bruno Elevation',
            #                   autosize=False,
            #                   # xaxis=dict(
            #                   #     tickmode='auto',
            #                   #     nticks=len(source_locations),
            #                   # ),
            #                   # yaxis=dict(
            #                   #     tickmode='auto',
            #                   #     nticks=len(release_names),
            #                   # ),
            #                   width=1200, height=1200,
            #                   margin=dict(l=65, r=50, b=65, t=90))
            # fig.show()
        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    scan_period_days = 14  # days
    # before_datetime = datetime.datetime.utcnow()
    # TODO: REMOVE fixed date after testing
    before_datetime = datetime.datetime(2023, 8, 1, 0, 0)  # There was a regression for GCP and Azure on 7/20
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

    commits_ordinals: Dict[str, Tuple[int, datetime.datetime]] = dict()
    # Create a dict mapping each commit to an increasing integer. This will allow more
    # efficient lookup later when we need to know whether one commit came after another.
    count = 0
    for row in commits_info.itertuples():
        commits_ordinals[row.commit] = (count, row.created_at)
        count += 1

    for test_id_suffix in LIMIT_TEST_ID_SUFFIXES:
        suffixed_records = f'''
            WITH junit_all AS(
                
                # Find 4.14 prowjobs which tested payloads during the scan period. For each
                # payload, aggregate the commits it included into an array.
                WITH payload_components AS(               
                    # Find all prowjobs which have run against a 4.14 payload commit
                    # in the last two months. 
                    SELECT  prowjob_build_id as pjbi, 
                            ARRAY_AGG(tag_source_location) as source_locations, 
                            ARRAY_AGG(tag_commit_id) as commits, 
                            ANY_VALUE(release_name) as release_name, 
                            ANY_VALUE(release_created) as release_created 
                    FROM openshift-gce-devel.ci_analysis_us.job_releases jr
                    WHERE   release_created {scan_period}
                            AND (release_name LIKE "4.14.0-0.nightly%" OR release_name LIKE "4.14.0-0.ci%")   
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
                        "N/A" as pr_sha
                FROM    `openshift-gce-devel.ci_analysis_us.junit` junit 
                        INNER JOIN payload_components ON junit.prowjob_build_id = payload_components.pjbi 
                WHERE   arch LIKE "{LIMIT_ARCH}"
                        AND network LIKE "{LIMIT_NETWORK}"
                        AND junit.modified_time {scan_period}
                        AND ENDS_WITH(test_id, "{test_id_suffix}")
                        AND test_name NOT LIKE "%disruption%"
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
                        jobs.pr_sha AS pr_sha
                FROM    `openshift-gce-devel.ci_analysis_us.junit_pr` junit_pr 
                        INNER JOIN payload_components ON junit_pr.prowjob_build_id = payload_components.pjbi
                        INNER JOIN `openshift-gce-devel.ci_analysis_us.jobs` jobs ON junit_pr.prowjob_build_id = jobs.prowjob_build_id 
                WHERE   arch LIKE "{LIMIT_ARCH}"
                        AND network LIKE "{LIMIT_NETWORK}"
                        AND junit_pr.modified_time {scan_period}
                        AND ENDS_WITH(test_id, "{test_id_suffix}") 
                        AND test_name NOT LIKE "%disruption%" 
                        AND platform LIKE "{LIMIT_PLATFORM}" 
                        AND upgrade LIKE "{LIMIT_UPGRADE}" 
            )
    
            SELECT  *
            FROM junit_all
            ORDER BY modified_time ASC
    '''

        # Construct a definitive set including all commits that ultimately merged during
        # our observation period.
        merged_commits = set(commits_info['commit'].tolist())

        print(f'Gathering test runs for suffix: {test_id_suffix}')
        all_records = main_client.query(suffixed_records).to_dataframe(create_bqstorage_client=True, progress_bar_type='tqdm')

        for row in all_records[all_records['is_pr'] == 0].itertuples():
            # if a test ran outside of pre-merge testing, we can assume that
            # any commits in the payload have previously merged (before our
            # observation period. Add any such commits to the merged_commits
            # set.
            merged_commits.update(row.commits)

        all_record_count = len(all_records.index)
        pr_record_count = len(all_records[all_records['is_pr'] == 1])
        print(f'Non-PR Records {all_record_count-pr_record_count}')
        print(f'    PR Records {pr_record_count}')

        if INCLUDE_PR_TESTS:
            # If a pr triggered test tests commits that are now a subset of all trusted / merged
            # commits, then promote the pr test to a non-pr test.
            for i, row in all_records.iterrows():
                if row.is_pr == 1 and row.pr_sha in merged_commits:
                    all_records.at[i, 'is_pr'] = 0

        # Drop all records that were pre-merge tests containing untrusted commits.
        trusted_records = all_records[all_records['is_pr'] == 0]
        trusted_record_count = len(trusted_records.index)
        print(f'Dropped {all_record_count-trusted_record_count} untrusted records')

        print(f'There are {len(trusted_records)} records to process with suffix: {test_id_suffix}')
        grouped_by_test_id = trusted_records.groupby('test_id', sort=False)
        pool = multiprocessing.Pool(processes=16)
        for _ in tqdm.tqdm(pool.imap_unordered(analyze_test_id, zip(grouped_by_test_id, itertools.repeat(commits_ordinals))), total=len(grouped_by_test_id.groups)):
            pass
        pool.close()
        time.sleep(10)

