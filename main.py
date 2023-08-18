#!/usr/bin/env python3
import multiprocessing
import datetime
import traceback
from typing import NamedTuple, List, Dict, Set, Tuple, Optional
from enum import Enum
import time

import numpy
import tqdm
from collections import OrderedDict

import itertools
import pandas
from google.cloud import bigquery
from fast_fisher import fast_fisher_cython
from functools import cached_property

import plotly.graph_objects as go


class Mode(Enum):
    DETECT_CURRENT = 0  # Prefer new test results against a commit to old.
    ANALYZE_PAST = 1


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

        self.behind_records: Optional[pandas.DataFrame] = None

        self.ahead_records: Optional[pandas.DataFrame] = None

        self.ahead_records_chron: Optional[pandas.DataFrame] = None  # will be set to ahead records, but secondary sort of time run time.

        self.discrete_outcomes: Optional[OutcomePair] = None

        self.release_streams: Dict[PayloadStreams, OrderedDict[str, bool]] = {
            PayloadStreams.NIGHTLY_PAYLOAD: OrderedDict(),  # We really just want an Ordered Set. Value doesn't matter.
            PayloadStreams.CI_PAYLOAD: OrderedDict(),
            PayloadStreams.PR_PAYLOAD: OrderedDict()
        }

    def set_data(self, behind_records: pandas.DataFrame, ahead_records: pandas.DataFrame):
        self.behind_records = behind_records.loc[behind_records['source_locations'] == self.source_location]
        self.behind_records = self.behind_records.reset_index(drop=True)  # Order records 0... so we can lop off after a specific index.
        # Note that for behind_records, tests are ordered in order of reverse git merge.
        first_behind_index_of_commit = self.behind_records.commits.where(self.behind_records.commits == self.commit_id).last_valid_index()
        self.behind_records = self.behind_records.iloc[first_behind_index_of_commit+1:]  # We don't want to our tests results or those of our children in the behind view

        self.ahead_records = ahead_records.loc[ahead_records['source_locations'] == self.source_location]
        self.ahead_records = self.ahead_records.reset_index(drop=True)  # Order records 0... so we can lop off before a specific index.
        first_ahead_index_of_commit = self.ahead_records.commits.where(self.ahead_records.commits == self.commit_id).first_valid_index()
        self.ahead_records = self.ahead_records.iloc[first_ahead_index_of_commit:]  # We don't need to keep any test records from before we arrived for look ahead

        tests_against_this_commit = self.ahead_records[self.ahead_records['commits'] == self.commit_id]
        self.discrete_outcomes = self.outcome_sums(tests_against_this_commit)
        self.ahead_records_chron = self.ahead_records.sort_values(by=['commits', 'modified_time'], ascending=[True, True], inplace=False)

        if self.created_at is None:
            # If we don't know when the commit has created using github information, then
            # approximate by the first release to test the commit.
            self.created_at = pandas.Timestamp(self.ahead_records_chron[self.ahead_records_chron['commits'] == self.commit_id].iloc[0].release_created, tz='UTC')
            pass

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
            success_count = selection['success_val'].sum()
            flake_count = selection['flake_count'].sum()
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

    def ahead(self, count: int, before_date: Optional[numpy.datetime64] = None) -> pandas.DataFrame:
        outcomes: Optional[pandas.DataFrame] = None

        if before_date:
            # If before_date is being used, the caller cares about specific time periods, so we
            # use ahead_records_chron. Unlike normal ahead calls, this means we will not
            # prefer new test results over old.
            outcomes = self.ahead_records_chron[(self.ahead_records_chron['modified_time'] < before_date)]
            if count > -1:
                if len(outcomes) >= count:
                    outcomes = outcomes[:count]
                else:
                    # Not enough results before date, ignore date.
                    outcomes = self.ahead_records_chron.iloc[:count]

        if outcomes is None:
            outcomes = self.ahead_records.iloc[:count]

        return outcomes

    def ahead_outcome(self, count: int, before_date: Optional[numpy.datetime64] = None) -> OutcomePair:
        """
        :param count: The desired number of test results to analyze. -1 for unlimited.
        :param before_date: Constrain to results before date. Unless count >-1 is specified in which case if there are
                            too few results before the date.
        :return:
        """
        return self.outcome_sums(self.ahead(count=count, before_date=before_date))

    def behind_outcome(self, count: Optional[int] = None, after_date: Optional[numpy.datetime64] = None) -> OutcomePair:
        if count is None and after_date is None:
            count = 10

        if count:
            filtered = self.behind_records.iloc[:count]
            return self.outcome_sums(filtered)
        else:
            filtered = self.behind_records[self.behind_records['modified_time'] > after_date]
            return self.outcome_sums(filtered)

    @cached_property
    def fe10(self):
        return self.ahead_outcome(count=10).fishers_exact(self.behind_outcome(count=10))

    @cached_property
    def mlog10p_10(self):
        return self.ahead_outcome(count=10).mlog10Test1t(self.behind_outcome(count=10))

    @cached_property
    def fe10_incoming_slope(self):
        if self.parent:
            return self.fe10 - self.parent.fe10
        return 0.0

    @cached_property
    def fe20(self):
        return self.ahead_outcome(count=20).fishers_exact(self.behind_outcome(count=20))

    @cached_property
    def mlog10p_20(self):
        return self.ahead_outcome(count=20).mlog10Test1t(self.behind_outcome(count=20))

    @cached_property
    def fe30(self):
        return self.ahead_outcome(count=30).fishers_exact(self.behind_outcome(count=30))

    @cached_property
    def mlog10p_30(self):
        return self.ahead_outcome(count=30).mlog10Test1t(self.behind_outcome(count=30))

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


OPERATING_MODE = Mode.ANALYZE_PAST
INCLUDE_PR_TESTS = True
LIMIT_ARCH = 'amd64'
LIMIT_NETWORK = 'ovn'
LIMIT_PLATFORM = 'azure'
LIMIT_UPGRADE = 'upgrade-micro'
# LIMIT_TEST_ID_SUFFIXES = list('abcdef0123456789')  # ids end with a hex digit, so cover everything.
# LIMIT_TEST_ID_SUFFIXES = list('56789')  # ids end with a hex digit, so cover everything.
LIMIT_TEST_ID_SUFFIXES = ['9d46f2845cf09db01147b356db9bfe0d']


def analyze_test_id(name_group_commits):
    name_group, commits_ordinals = name_group_commits
    name, test_id_group = name_group
    grouped_by_nurp = test_id_group.groupby(['network', 'upgrade', 'arch', 'platform', 'test_id'], sort=False)
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
                    created_at = commits_ordinals.get(commit_id, (None,None))[1]
                    new_commit = CommitOutcomes(source_locations[idx], commit_id, created_at=created_at)
                    all_commits[new_commit.commit_id] = new_commit
                all_commits[commit_id].record_observation_in_release(t.release_name)

        # There is no guarantee that the order we observe commits being tested is the
        # order in which they merged. Build a definitive order using information from github.
        ordered_commits: List[str] = list(all_commits.keys())

        # If the exact ordering of a commit is not known, preserve
        # the ordering encountered in the incoming list.
        pos_for_missing_info = -100000000000

        def ordinal_for_commit(commit_id):
            nonlocal pos_for_missing_info
            ordinal = commits_ordinals.get(commit_id, (pos_for_missing_info,))
            if ordinal[0] < 0:
                pos_for_missing_info += 1
            else:
                pass
            return ordinal

        ordered_commits.sort(key=ordinal_for_commit)

        linked: Set[str] = set()
        linked_commits: Dict[SourceLocation, CommitOutcomes] = dict()
        for commit_id in ordered_commits:
            commit_to_update = all_commits[commit_id]
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

        # We are about to process test results for self+ancestors. See long comment
        # above about why commits are ascending order (order of commit history within
        # a given repo) and modified_time (test run time) is descending (reverse
        # chronological).
        flattened_nurp_group.sort_values(by=['commits', 'modified_time'], ascending=[True, OPERATING_MODE == Mode.ANALYZE_PAST], inplace=True)

        # Within a release payload, a commit may be encountered multiple times: one
        # for each component it is associated with (e.g. openshift/oc is associated with
        # cli, cli-artifacts, deployer, and tools). We don't want each these components
        # count an individual success/failure against the oc commit, or we will
        # 4x count it. Convert the commits into a set to dedupe.
        flattened_nurp_group.drop_duplicates(inplace=True)
        ahead_test_records = flattened_nurp_group

        # We are about to process test results for children. See long comment
        # above about why commits are descending order (reverse of commit history within
        # a given repo) and modified_time (test run time) is descending (reverse
        # chronological).
        behind_test_records = flattened_nurp_group.sort_values(by=['commits', 'modified_time'], ascending=[False, OPERATING_MODE == Mode.ANALYZE_PAST], inplace=False)

        for commit_outcome in all_commits.values():
            commit_outcome.set_data(behind_test_records, ahead_test_records)

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
        unresolved_regression: List[CommitOutcomes] = list()
        by_le_grande_order: List[CommitOutcomes] = sorted(list(all_commits.values()), key=le_grande_order)
        for c in by_le_grande_order:
            # if c.fe_dynamic > -0.98:
            #     break
            # if not c.is_valley():
            #     continue
            relevant_count += 1
            print(f'{le_grande_order(c)}')
            # regressed, resolution = c.regression_info()
            # if regressed and not resolution:
            #     unresolved_regression.append(c)
            print(c)
            print()

        print(f'Found {relevant_count}')
        print(f'Found {len(unresolved_regression)} unresolved regressions')
        for c in unresolved_regression:
            print(c)
            print()

        z = []
        release_names: OrderedDict[str, bool] = OrderedDict()  # y axis
        source_locations = None  # x axis. It is assumed that the source_locations will not change in a given stream over the span of our results
        for t in nurp_group.itertuples():
            if PayloadStreams.get_stream(t.release_name) != PayloadStreams.NIGHTLY_PAYLOAD:
                continue

            release_names[t.release_name] = True
            if not source_locations:
                source_locations = sorted(list(t.source_locations))

            commit_outcomes: List[CommitOutcomes] = list()
            for commit_id in t.commits:
                commit_outcomes.append(all_commits[commit_id])

            # We need to take care to tolerate if there are new / missing source locations,
            # but this should be exceedingly rare.
            commit_outcomes_by_source_location = {commit.source_location: commit for commit in commit_outcomes}

            z_entry = [t.release_name]
            for source_location in source_locations:
                c: CommitOutcomes = commit_outcomes_by_source_location.get(source_location, None)
                if c:
                    #z_entry.append(c.mlog10p_10)
                    z_entry.append(f'g{c.commit_id[:5]}:t{c.mlog10p_10}')
                else:
                    #z_entry.append(0.0)
                    z_entry.append('?')

            z.append(z_entry)

        pandas.DataFrame(z).to_csv('what.csv', index_label=source_locations)


        try:
            fig = go.Figure(data=[go.Surface(z=z, x=source_locations, y=list(release_names.keys()))])
            fig.update_layout(title='Mt Bruno Elevation',
                              autosize=False,
                              xaxis=dict(
                                  tickmode='auto',
                                  nticks=len(source_locations),
                              ),
                              yaxis=dict(
                                  tickmode='auto',
                                  nticks=len(release_names),
                              ),
                              width=1200, height=1200,
                              margin=dict(l=65, r=50, b=65, t=90))
            fig.show()
        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    scan_period_days = 14  # days
    # before_datetime = datetime.datetime.utcnow()
    # TODO: REMOVE fixed date after testing
    before_datetime = datetime.datetime(2023, 8, 1, 0, 0)  # There was a regression for GCP and Azure on 7/20
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

