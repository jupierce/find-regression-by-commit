#!/usr/bin/env python3
import multiprocessing
import datetime
from typing import NamedTuple, List, Dict
import time
import tqdm

import itertools
import pandas
from google.cloud import bigquery
from fast_fisher import fast_fisher_cython
import matplotlib.pyplot as plt
import pathlib


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


class ResultSum:

    def __init__(self, max_records: int, success_count=0, failure_count=0):
        self.max_records = max_records
        self.success_count = success_count
        self.failure_count = failure_count

    def add_success(self):
        if self.success_count + self.failure_count < self.max_records:
            self.success_count += 1

    def add_failure(self, amount=1):
        if self.success_count + self.failure_count < self.max_records:
            self.failure_count += amount
            if self.failure_count < 0:
                # This can happen if we hit a flake_count > row, but datetime aligns us after the rows of the failures they account for.
                self.failure_count = 0

    def pass_rate(self) -> float:
        if self.success_count > 0 or self.failure_count > 0:
            return self.success_count / (self.success_count + self.failure_count)
        return 0.0

    def fishers_exact_regressed(self, result_sum2) -> float:
        # If at least ten tests have run including this commit and
        # if we have regressed at least 10% relative to the result we are comparing to.
        if result_sum2.pass_rate() - self.pass_rate() > 0.10:
            return 1 - fast_fisher_cython.fisher_exact(
                self.failure_count, self.success_count,
                result_sum2.failure_count, result_sum2.success_count,
                alternative='greater')
        return 0.0

    def __str__(self):
        return f'[s={self.success_count} f={self.failure_count} r={int(self.pass_rate() * 100)}%]'


MAX_WINDOW_SIZE = 30


class CommitSums:

    def __init__(self, source_location:str, commit_id: str, parent_commit_sums=None, child_commit_sums=None):
        self.source_location = source_location
        self.commit_id = commit_id
        self.success_count = 0
        self.failure_count = 0
        self.flake_count = 0
        self.parent = parent_commit_sums
        self.child = child_commit_sums

        self.ahead_10: ResultSum = ResultSum(10)
        self.ahead_20: ResultSum = ResultSum(20)
        self.ahead_30: ResultSum = ResultSum(MAX_WINDOW_SIZE)

        self.behind_10: ResultSum = ResultSum(10)
        self.behind_20: ResultSum = ResultSum(20)
        self.behind_30: ResultSum = ResultSum(MAX_WINDOW_SIZE)

    def add_success(self):

        # Inform children of a success behind them
        node = self.child
        count = 0
        while node is not None and count <= MAX_WINDOW_SIZE:
            node.behind_10.add_success()
            node.behind_20.add_success()
            node.behind_30.add_success()
            node = node.child
            count += 1

        # Inform parents and self of a success ahead of them
        node = self
        count = 0
        while node is not None and count <= MAX_WINDOW_SIZE:
            node.ahead_10.add_success()
            node.ahead_20.add_success()
            node.ahead_30.add_success()
            node = node.parent
            count += 1

    def add_failure(self, amount=1):
        if amount == 0:  # possible when decrementing flake_count and flake_count=0
            return

        # Inform children of a failure behind them
        node = self.child
        count = 0
        while node is not None and count <= MAX_WINDOW_SIZE:
            node.behind_10.add_failure(amount)
            node.behind_20.add_failure(amount)
            node.behind_30.add_failure(amount)
            node = node.child
            count += 1

        # Inform parents and self of a failure ahead of them
        node = self
        count = 0
        while node is not None and count <= MAX_WINDOW_SIZE:
            node.ahead_10.add_failure(amount)
            node.ahead_20.add_failure(amount)
            node.ahead_30.add_failure(amount)
            node = node.parent
            count += 1

    def fe10(self):
        return self.ahead_10.fishers_exact_regressed(self.behind_10)

    def fe20(self):
        return self.ahead_20.fishers_exact_regressed(self.behind_20)

    def fe30(self):
        return self.ahead_30.fishers_exact_regressed(self.behind_30)

    def __str__(self):
        v = f'''Commit: {self.source_location}/commit/{self.commit_id}
10:
  behind10: {self.behind_10}        
  ahead10: {self.ahead_10}
  fe10: {self.fe10()}
20:
  behind10: {self.behind_20}        
  ahead10: {self.ahead_20}
  fe20: {self.fe20()}
30:
  behind30: {self.behind_30}        
  ahead30: {self.ahead_30}
  fe30: {self.fe30()}
'''
        if self.parent:
            v += f'Parent: {self.parent.commit_id}\n'
        if self.child:
            v += f'Child: {self.child.commit_id}\n'
        return v


SourceLocation = str
CommitId = str

SCAN_PERIOD = 'BETWEEN DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 2 WEEK) AND CURRENT_DATETIME()'
# SCAN_PERIOD = 'BETWEEN DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 1 MONTH) AND CURRENT_DATETIME()'
LIMIT_ARCH = 'amd64'
LIMIT_NETWORK = 'ovn'
LIMIT_PLATFORM = 'gcp'
LIMIT_TEST_ID = None
# LIMIT_TEST_ID = ['openshift-tests:cb921b4a3fa31e83daa90cc418bb1cbc']


def analyze_test_id(name_group_commits):
    name_group, commits_ordinals = name_group_commits
    name, test_id_group = name_group
    grouped_by_nurp = test_id_group.groupby(['network', 'upgrade', 'arch', 'platform', 'test_id'], sort=False)
    for name, nurp_group in grouped_by_nurp:
        # print(f'Processing {name}')

        linked_commits: Dict[SourceLocation, CommitSums] = dict()
        all_commits: Dict[CommitId, CommitSums] = dict()

        first_row = nurp_group.iloc[0]
        test_name = first_row['test_name']
        qtest_id = f"{first_row['network']}_{first_row['upgrade']}_{first_row['arch']}_{first_row['platform']}_{first_row['test_id']}"

        pos_for_missing_info = -100000000000
        for t in nurp_group.itertuples():
            source_locations = t.source_locations
            commits = list(t.commits)

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
            for idx in range(len(commits)):
                source_location = source_locations[idx]
                commit_id = commits[idx]
                if source_location not in linked_commits:
                    new_commit = CommitSums(source_location, commit_id)  # Initial commit found for repo
                    linked_commits[source_location] = new_commit
                    all_commits[new_commit.commit_id] = new_commit
                else:
                    if commit_id not in all_commits:
                        new_commit = CommitSums(source_location, commit_id,
                                                parent_commit_sums=linked_commits[source_location])
                        linked_commits[source_location].child = new_commit
                        linked_commits[source_location] = new_commit
                        all_commits[new_commit.commit_id] = new_commit

        # Iterate through again to cuckoo the test outcomes back and forth
        # through the commit graph for each repo.
        for t in nurp_group.itertuples():
            source_locations = t.source_locations
            commits = t.commits
            for idx in range(len(commits)):
                commit_id = commits[idx]
                target_commit = all_commits[commit_id]
                if t.success_val == 1:
                    target_commit.add_success()
                else:
                    target_commit.add_failure()
                target_commit.add_failure(-1 * t.flake_count)

        group_frame = pandas.DataFrame(columns=[
            'release_name',
            'modified_time',
            'tag_commit_id',
            'link',
            'fe10',
            'fe20',
            'fe30',
            'test_id',
            'test_name',
        ])
        gf_idx = 0

        for t in nurp_group[nurp_group['release_name'].str.contains('nightly')].itertuples():
            source_locations = t.source_locations
            commits = t.commits
            for idx in range(len(commits)):
                commit_id = commits[idx]
                if commit_id in all_commits:
                    source_location = source_locations[idx]
                    target_commit = all_commits[commit_id]
                    group_frame.loc[gf_idx] = [
                        t.release_name,
                        t.modified_time,
                        commit_id,
                        f'{source_location}/commit/{commit_id}',
                        target_commit.fe10(),
                        target_commit.fe20(),
                        target_commit.fe30(),
                        qtest_id,
                        test_name
                    ]
                    gf_idx += 1
                    del all_commits[commit_id]  # Don't add this commit to table again

        by_mod = group_frame.groupby('release_name').aggregate(
            {
                'release_name': 'min',
                'test_name': 'min',
                'test_id': 'min',
                'fe10': 'mean',
                'fe20': 'mean',
                'fe30': 'mean',
            }
        )
        if len(by_mod.loc[(by_mod['fe10'] > 0.95) | (by_mod['fe20'] > 0.95) | (
                by_mod['fe30'] > 0.95)].index) > 1:
            group_frame['fe10'] = group_frame['fe10'].apply(lambda x: x * 3)
            group_frame['fe20'] = group_frame['fe20'].apply(lambda x: x * 2)
            pathlib.Path('tests').mkdir(parents=True, exist_ok=True)
            group_frame.to_csv(f'tests/{qtest_id}.csv')


if __name__ == '__main__':
    scan_period_days = 14  # days
    before_datetime = datetime.datetime.utcnow() - datetime.timedelta(days=0.5)  # TODO: github archive does not immediately have day when UTC changes over to new day; find better commit history method.
    after_datetime = before_datetime - datetime.timedelta(days=scan_period_days)
    before_str = before_datetime.strftime("%Y-%m-%d %H:%M:%S") + '+00'
    after_str = after_datetime.strftime("%Y-%m-%d %H:%M:%S") + '+00'

    main_client = bigquery.Client(project='openshift-gce-devel')

    find_commits = ''
    date_stepper = before_datetime
    for i in range(scan_period_days+1):

        if find_commits:
            find_commits += '\nUNION ALL\n'

        find_commits += f'''
        SELECT created_at, CONCAT("github.com/", repo.name) as repo, JSON_VALUE(payload,'$.head' ) as head, JSON_VALUE(payload,'$.before' ) as before
        FROM `githubarchive.day.{date_stepper.strftime("%Y%m%d")}`
        WHERE
            type = "PushEvent"
            AND (repo.name LIKE "operator-framework/%" OR repo.name LIKE "openshift/%") 
        '''
        date_stepper = date_stepper - datetime.timedelta(days=1)

    find_commits = f'''
        WITH commits AS (
            {find_commits}
        )   SELECT MIN(created_at) as created_at, head, ANY_VALUE(before) as before
            FROM commits
            GROUP BY head 
            ORDER BY created_at ASC
    '''

    commits_info = main_client.query(find_commits).to_dataframe(create_bqstorage_client=True, progress_bar_type='tqdm')
    commits_ordinals: Dict[str, int] = dict()
    # Create a dict mapping each commit to an increasing integer. This will allow more
    # efficient lookup later when we need to know whether one commit came after another.
    count = 0
    for row in commits_info.itertuples():
        commits_ordinals[row.head] = count
        count += 1

    for test_id_suffix in list('d'):  # TODO restore: list('abcdef0123456789'):
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
                    WHERE   release_created {SCAN_PERIOD}
                            # TODO: THIS ONLY INCLUDES NIGHTLIES AT PRESENT. INCLUDE CI PAYLOADS LATER
                            AND (release_name LIKE "4.14.0-0.nightly%" OR release_name LIKE "4.14.0-0.ci%")   
                            # AND release_name LIKE "4.14.0-0.nightly%"
                    GROUP BY prowjob_build_id
                )
    
                # Find all junit tests run in non-PR triggered tests which ran as part of those prowjobs over the last two months
                SELECT  *
                FROM `openshift-gce-devel.ci_analysis_us.junit` junit INNER JOIN payload_components ON junit.prowjob_build_id = payload_components.pjbi 
                WHERE   arch LIKE "{LIMIT_ARCH}"
                        AND network LIKE "{LIMIT_NETWORK}"
                        AND junit.modified_time {SCAN_PERIOD}
                        AND ENDS_WITH(test_id, "{test_id_suffix}")
                        AND test_name NOT LIKE "%disruption%"
                        AND platform LIKE "{LIMIT_PLATFORM}" 
                # IGNORE TESTING FROM PRs for the time being.
                # UNION ALL    
    
                # # Find all junit tests run in PR triggered tests which ran as part of those prowjobs over the last two months
                # SELECT  *
                # FROM `openshift-gce-devel.ci_analysis_us.junit_pr` junit_pr INNER JOIN payload_components ON junit_pr.prowjob_build_id = payload_components.pjbi 
                # WHERE   arch LIKE "{LIMIT_ARCH}"
                #         AND network LIKE "{LIMIT_NETWORK}"
                #         AND junit_pr.modified_time {SCAN_PERIOD}
                #         AND ENDS_WITH(test_id, "{test_id_suffix}") 
                #         AND test_name NOT LIKE "%disruption%" 
                #         AND platform LIKE "{LIMIT_PLATFORM}" 
            )
    
            SELECT  *
            FROM junit_all
            ORDER BY modified_time ASC
    '''
        all_records = main_client.query(suffixed_records).to_dataframe(create_bqstorage_client=True, progress_bar_type='tqdm')
        print(f'There are {len(all_records)} records to process with suffix: {test_id_suffix}')
        grouped_by_test_id = all_records.groupby('test_id')
        pool = multiprocessing.Pool(processes=16)
        for _ in tqdm.tqdm(pool.imap_unordered(analyze_test_id, zip(grouped_by_test_id, itertools.repeat(commits_ordinals))), total=len(grouped_by_test_id.groups)):
            pass
        pool.close()
        time.sleep(10)

