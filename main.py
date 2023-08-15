#!/usr/bin/env python3
import multiprocessing
import datetime
from typing import NamedTuple, List, Dict, Set, Tuple
import time
import tqdm
from collections import OrderedDict

import itertools
import pandas
from google.cloud import bigquery
from fast_fisher import fast_fisher_cython
from functools import cached_property

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


class OutcomePair:

    def __init__(self, max_records: int, commit_id):
        self.max_records = max_records
        self.success_count = 0
        self.failure_count = 0
        self.commit_id = commit_id
        self.prowjob_urls: OrderedDict = OrderedDict()

    def add_success(self, amount=1, prowjob_url: str = None):
        if self.max_records == -1 or self.success_count + self.failure_count < self.max_records:
            self.success_count += amount
            if prowjob_url:
                self.prowjob_urls[prowjob_url] = 'S'

    def add_failure(self, amount=1, prowjob_url: str = None):
        if self.max_records == -1 or self.success_count + self.failure_count < self.max_records:
            self.failure_count += amount
            if self.failure_count < 0:
                # This can happen if we hit a flake_count > row, but datetime aligns us after the rows of the failures they account for.
                self.failure_count = 0
                if prowjob_url:
                    self.prowjob_urls[prowjob_url] = 'F'

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

    def __str__(self):
        return f'[s={self.success_count} f={self.failure_count} r={int(self.pass_rate() * 100)}%]'


MAX_WINDOW_SIZE = 30


class CommitOutcomes:

    def __init__(self, source_location: str, commit_id: str, parent_commit_sums=None, child_commit_sums=None):
        self.source_location = source_location
        self.commit_id = commit_id
        self.parent = parent_commit_sums
        self.child = child_commit_sums

        self.discrete_outcomes = OutcomePair(-1, commit_id)  # Tests run against exactly this commit.

        self.ahead_10: OutcomePair = OutcomePair(10, commit_id)
        self.ahead_20: OutcomePair = OutcomePair(20, commit_id)
        self.ahead_30: OutcomePair = OutcomePair(30, commit_id)

        self.behind_10: OutcomePair = OutcomePair(10, commit_id)
        self.behind_20: OutcomePair = OutcomePair(20, commit_id)
        self.behind_30: OutcomePair = OutcomePair(30, commit_id)

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

    def inform_children_of_success_behind_them(self, prowjob_url: str = None):
        # Inform children of a success behind them
        node = self.child
        count = 0
        while node is not None and count <= MAX_WINDOW_SIZE:
            node.behind_10.add_success(prowjob_url=prowjob_url)
            node.behind_20.add_success(prowjob_url=prowjob_url)
            node.behind_30.add_success(prowjob_url=prowjob_url)
            node = node.child
            count += 1

    def inform_ancestors_of_success_including_them(self, prowjob_url: str = None):
        # Inform parents and self of a success ahead of them
        node = self
        self.discrete_outcomes.add_success(prowjob_url=prowjob_url)
        count = 0
        while node is not None and count <= MAX_WINDOW_SIZE:
            node.ahead_10.add_success(prowjob_url=prowjob_url)
            node.ahead_20.add_success(prowjob_url=prowjob_url)
            node.ahead_30.add_success(prowjob_url=prowjob_url)
            node = node.parent
            count += 1

    def inform_children_of_failure_behind_them(self, amount=1, prowjob_url: str = None):
        if amount == 0:  # possible when decrementing flake_count and flake_count=0
            return

        # Inform children of a failure behind them
        node = self.child
        count = 0
        while node is not None and count <= MAX_WINDOW_SIZE:
            node.behind_10.add_failure(amount, prowjob_url=prowjob_url)
            node.behind_20.add_failure(amount, prowjob_url=prowjob_url)
            node.behind_30.add_failure(amount, prowjob_url=prowjob_url)
            node = node.child
            count += 1

    def inform_ancestors_of_failure_including_them(self, amount=1, prowjob_url: str = None):
        # Inform parents and self of a failure ahead of them
        node = self
        self.discrete_outcomes.add_failure(amount=amount, prowjob_url=prowjob_url)
        count = 0
        while node is not None and count <= MAX_WINDOW_SIZE:
            node.ahead_10.add_failure(amount, prowjob_url=prowjob_url)
            node.ahead_20.add_failure(amount, prowjob_url=prowjob_url)
            node.ahead_30.add_failure(amount, prowjob_url=prowjob_url)
            node = node.parent
            count += 1

    @cached_property
    def fe10(self):
        return self.ahead_10.fishers_exact(self.behind_10)

    @cached_property
    def fe10_incoming_slope(self):
        if self.parent:
            return self.fe10 - self.parent.fe10
        return 0.0

    @cached_property
    def fe20(self):
        return self.ahead_20.fishers_exact(self.behind_20)

    @cached_property
    def fe30(self):
        return self.ahead_30.fishers_exact(self.behind_30)

    @cached_property
    def worse_than_parent(self) -> bool:
        return self.parent and self.parent.fe10 > self.fe10

    def _behind_scan(self) -> Tuple[int, int]:
        successes, failures = self.discrete_outcomes.success_count, self.discrete_outcomes.failure_count
        if self.worse_than_parent:
            # If I have a parent and their fe10 is better than or
            # equal to my own.
            p_successes, p_failures = self.parent._behind_scan()
            successes += p_successes
            failures += p_failures
        return successes, failures

    @cached_property
    def behind_dynamic(self) -> OutcomePair:
        bs = self._behind_scan()
        behind = OutcomePair(-1, self.commit_id)
        # Looking behind us, we should not include our own successes/failures
        behind.add_success(amount=bs[0] - self.discrete_outcomes.success_count)
        behind.add_failure(amount=bs[1] - self.discrete_outcomes.failure_count)
        return behind

    @cached_property
    def worse_than_child(self) -> bool:
        # if a commit does not have a child, it is the last known commit
        # for a repo. Return True for childless so that it can be considered
        # a peak.
        # Otherwise, return True only if the child fe is better than ours.
        return (not self.child) or self.child.fe10 > self.fe10

    def _ahead_scan(self) -> Tuple[int, int]:
        successes, failures = self.discrete_outcomes.success_count, self.discrete_outcomes.failure_count

        # If I have a child and their fe10 is better than mine, failures are tilted in
        # this commit's direction. If child.fe10 happens to be equal
        # to our own, we reason that the child failures are statistically
        # worse because they include our own failures but have not
        # begun to normalize. Thus the child is more likely to be the peak.
        if self.worse_than_child:
            if self.child and self.child.fe10 <= 0:
                # If the child also has a decline in signal, their results are
                # going to be included in our dynamically sized fe.
                c_successes, c_failures = self.child._ahead_scan()
                successes += c_successes
                failures += c_failures
        return successes, failures

    @cached_property
    def ahead_dynamic(self) -> OutcomePair:
        successes, failures = self._ahead_scan()
        ahead = OutcomePair(-1, commit_id=self.commit_id)
        ahead.add_success(amount=successes)
        ahead.add_failure(amount=failures)
        return ahead

    def is_peak(self):
        return self.worse_than_parent and self.worse_than_child

    @cached_property
    def fe_dynamic(self):
        behind = self.behind_dynamic
        ahead = self.ahead_dynamic
        return ahead.fishers_exact(behind)

    def __str__(self):
        v = f'''Commit: {self.source_location}/commit/{self.commit_id}
Peak: {self.is_peak()}
Discrete: {self.discrete_outcomes}
10:
  behind10: {self.behind_10}        
  ahead10: {self.ahead_10}
  fe10: {self.fe10}
20:
  behind10: {self.behind_20}        
  ahead10: {self.ahead_20}
  fe20: {self.fe20}
30:
  behind30: {self.behind_30}        
  ahead30: {self.ahead_30}
  fe30: {self.fe30}
dynamic:
  behind: {self.behind_dynamic}
  ahead: {self.ahead_dynamic}
  fe: {self.fe_dynamic}
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
LIMIT_NETWORK = 'ovn'
LIMIT_PLATFORM = 'gcp'
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

        linked: Set[str] = set()
        linked_commits: Dict[SourceLocation, CommitOutcomes] = dict()
        all_commits: Dict[CommitId, CommitOutcomes] = dict()

        # Contains a list of all commits. Within a given repo, the commits in this list should
        # be in chronological order. Between different repos, there is no guarantee.
        ordered_commits = list()

        first_row = nurp_group.iloc[0]
        test_name = first_row['test_name']
        network = first_row['network']
        upgrade = first_row['upgrade']
        arch = first_row['arch']
        platform = first_row['platform']
        test_id = first_row['test_id']
        qtest_id = f"{network}_{upgrade}_{arch}_{platform}_{test_id}"

        pos_for_missing_info = -100000000000
        for t in nurp_group.itertuples():
            source_locations = t.source_locations
            commits = list(t.commits)

            # Associate the commit_ids and the source location for that
            # commit.
            for idx in range(len(commits)):
                commit_id = commits[idx]
                if commit_id not in all_commits:
                    new_commit = CommitOutcomes(source_locations[idx], commit_id)
                    all_commits[new_commit.commit_id] = new_commit

            # If the exact ordering of a commit is not known, preserve
            # the ordering encountered in the incoming list.
            def ordinal_for_commit(commit_id):
                nonlocal pos_for_missing_info
                ordinal = commits_ordinals.get(commit_id, pos_for_missing_info)
                if ordinal < 0:
                    pos_for_missing_info += 1
                else:
                    pass
                return ordinal

            commits.sort(key=ordinal_for_commit)
            ordered_commits.extend(commits)

            for idx in range(len(commits)):
                commit_id = commits[idx]
                new_commit = all_commits[commit_id]
                source_location = new_commit.source_location
                if source_location not in linked_commits:
                    # Initial commit found for repo
                    linked_commits[source_location] = new_commit
                    linked.add(commit_id)
                else:
                    if commit_id not in linked:
                        linked.add(commit_id)
                        new_commit.parent = linked_commits[source_location]
                        linked_commits[source_location].child = new_commit
                        linked_commits[source_location] = new_commit

            # with pathlib.Path(f'{qtest_id}-commits.txt').open(mode='w+') as f:
            #     for commit_sum in all_commits.values():
            #         f.write(f'Commit {commit_sum.commit_id} in {commit_sum.source_location} has parent commit: {commit_sum.get_parent_commit_id()}\n')
            #
            # nurp_group.to_csv(f'{qtest_id}-nurp-tests.csv')

        # Make a row for each test result against each commit.
        flattened_nurp_group = nurp_group.drop('source_locations', axis=1).explode('commits')  # pandas cannot drop duplicates with a column containing arrays. We don't need source locations in longer, so drop it.
        ordered_commits = list(dict.fromkeys(ordered_commits))  # fast way to remove dupes while preserving order: https://stackoverflow.com/a/17016257
        flattened_nurp_group['commits'] = pandas.Categorical(flattened_nurp_group['commits'], ordered_commits)  # When subsequently sorted, this will sort the entries by their location in the ordered_commits
        flattened_nurp_group.sort_values(by=['commits', 'modified_time'], inplace=True)

        # Within a release payload, a commit may be encountered multiple times: one
        # for each component it is associated with (e.g. openshift/oc is associated with
        # cli, cli-artifacts, deployer, and tools). We don't want each these components
        # count an individual success/failure against the oc commit, or we will
        # 4x count it. Convert the commits into a set to dedupe.
        flattened_nurp_group.drop_duplicates(inplace=True)

        # Iterate through again to cuckoo the test outcomes back and forth
        # through the commit graph for each repo.
        for t in flattened_nurp_group.itertuples():
            prowjob_name = t.prowjob_name
            prowjob_build_id = t.prowjob_build_id
            job_link = f'https://prow.ci.openshift.org/view/gs/origin-ci-test/logs/{prowjob_name}/{prowjob_build_id}'

            target_commit = all_commits[t.commits]
            if t.success_val == 1:
                target_commit.inform_ancestors_of_success_including_them(prowjob_url=job_link)
            else:
                target_commit.inform_ancestors_of_failure_including_them(prowjob_url=job_link)
            target_commit.inform_ancestors_of_failure_including_them(-1 * t.flake_count)


        # For the "behind" aggregators, we want to prioritize successes/failures
        # as close to the introduction of the commit as possible. Reversing the
        # nurp puts modified_time in DESC order (backwards in time).
        # Without this reversal, old ancestors will fill up a commit's
        # max aggregation value before newer/closer ancestor results.
        flattened_nurp_group_reversed = flattened_nurp_group.iloc[::-1]
        for t in flattened_nurp_group_reversed.itertuples():
            prowjob_name = t.prowjob_name
            prowjob_build_id = t.prowjob_build_id
            job_link = f'https://prow.ci.openshift.org/view/gs/origin-ci-test/logs/{prowjob_name}/{prowjob_build_id}'

            target_commit = all_commits[t.commits]
            if t.success_val == 1:
                target_commit.inform_children_of_success_behind_them(prowjob_url=job_link)
            else:
                target_commit.inform_children_of_failure_behind_them(prowjob_url=job_link)
            target_commit.inform_children_of_failure_behind_them(-1 * t.flake_count)

        def le_grande_order(commit: CommitOutcomes):
            # Returning a tuple means order by first attribute, then next, then..
            return (
                commit.fe_dynamic,
                commit.fe10_incoming_slope,
                commit.discrete_outcomes.pass_rate(),
                -1 * commit.ahead_dynamic.failure_count,
                commit.fe10,
                commit.fe20,
                commit.fe30,
            )

        relevant_count = 0
        by_le_grande_order = sorted(list(all_commits.values()), key=le_grande_order)
        for c in by_le_grande_order:
            if c.fe_dynamic > -0.98:
                break
            if not c.is_peak():
                continue
            relevant_count += 1
            print(f'{c.fe_dynamic} -> {le_grande_order(c)}')
            print(c)
            print()

        print(f'Found {relevant_count}')


if __name__ == '__main__':
    scan_period_days = 14  # days
    # before_datetime = datetime.datetime.utcnow()
    # TODO: REMOVE subtraction in before_datetime to prevent searching in the past
    before_datetime = datetime.datetime.utcnow() - datetime.timedelta(days=scan_period_days)
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

    commits_ordinals: Dict[str, int] = dict()
    # Create a dict mapping each commit to an increasing integer. This will allow more
    # efficient lookup later when we need to know whether one commit came after another.
    count = 0
    for row in commits_info.itertuples():
        commits_ordinals[row.commit] = count
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
                # IGNORE TESTING FROM PRs for the time being.
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

