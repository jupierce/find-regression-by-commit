#!/usr/bin/env python3

from collections import defaultdict
from typing import NamedTuple, List

import matplotlib.pyplot
from google.cloud import bigquery
from fast_fisher import fast_fisher_cython
import matplotlib.pyplot as plt
import pathlib


class WrappedInteger():
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


client = bigquery.Client(project='openshift-gce-devel')


class Coordinate(NamedTuple):
    network: str
    upgrade: str
    platform: str
    arch: str
    test_id: str

    def uid(self):
        return f'{self.network}-{self.upgrade}-{self.arch}-{self.platform}-{self.test_id}'


success_sums = defaultdict(WrappedInteger)
failure_sums = defaultdict(WrappedInteger)
flake_sums = defaultdict(WrappedInteger)


if __name__ == '__main__':
    # QUERY = 'SELECT * FROM `openshift-gce-devel.ci_analysis_us.tmp_results_by_commit` ORDER BY first_instance_time DESC'

    # Full granularity
#     QUERY = '''
# SELECT MIN(modified_time) as first_instance_time, junit.network, junit.upgrade, junit.platform, junit.arch, test_id, (COUNT(*)-SUM(success_val)-SUM(flake_count)) as fail_count, SUM(success_val) as success_count, SUM(flake_count) as flake_count, tag_source_location, tag_commit_id FROM `openshift-gce-devel.ci_analysis_us.junit` junit, `openshift-gce-devel.ci_analysis_us.job_releases` job_releases
# WHERE
# junit.modified_time BETWEEN DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 48 HOUR) AND CURRENT_DATETIME()
# AND job_releases.prowjob_build_id = junit.prowjob_build_id
# GROUP BY junit.network, junit.platform, junit.arch, junit.upgrade, test_id, job_releases.tag_source_location, job_releases.tag_commit_id
# ORDER BY first_instance_time DESC
#     '''

    # QUERY = '''
    # SELECT MIN(modified_time) as first_instance_time, "" as network, "" as upgrade, "" as platform, "" as arch, test_id, (COUNT(*)-SUM(success_val)-SUM(flake_count)) as fail_count, SUM(success_val) as success_count, SUM(flake_count) as flake_count, tag_source_location, tag_commit_id FROM `openshift-gce-devel.ci_analysis_us.junit` junit, `openshift-gce-devel.ci_analysis_us.job_releases` job_releases
    # WHERE
    # junit.modified_time BETWEEN DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 48 HOUR) AND CURRENT_DATETIME()
    # AND job_releases.tag_commit_id != ""
    # AND job_releases.prowjob_build_id = junit.prowjob_build_id
    # GROUP BY test_id, job_releases.tag_source_location, job_releases.tag_commit_id
    # ORDER BY first_instance_time DESC
    # '''

    # LIMITING TO AMD64 for now! See WHERE CLAUSE
    QUERY = '''
        WITH junit_all AS (
            WITH payload_components AS(
                WITH commits AS(
                    SELECT  created_at, 
                            JSON_VALUE(payload,'$.ref' ) as branch, 
                            CONCAT("github.com/", repo.name) as repo, 
                            JSON_VALUE(payload,'$.head' ) as head, 
                            JSON_VALUE(payload,'$.before' ) as before   
                    FROM `githubarchive.day.2*` 
                    WHERE   type = "PushEvent" 
                            AND (_TABLE_SUFFIX LIKE "02306%" or _TABLE_SUFFIX LIKE "02307%")
                            AND (repo.name LIKE "operator-framework/%" OR repo.name LIKE "openshift/%") 
                            AND ENDS_WITH(JSON_VALUE(payload,'$.ref' ), "/master")
                ) 
                SELECT  prowjob_build_id as pjbi, 
                        tag_source_location, 
                        tag_commit_id, 
                        ANY_VALUE(release_name), 
                        MIN(release_created) as first_release_date, 
                        created_at 
                FROM openshift-gce-devel.ci_analysis_us.job_releases jr JOIN commits ON commits.head = jr.tag_commit_id 
                GROUP BY prowjob_build_id, tag_source_location, tag_commit_id, created_at
            )
            SELECT  *
            FROM `openshift-gce-devel.ci_analysis_us.junit` junit CROSS JOIN payload_components 
            WHERE   junit.prowjob_build_id = payload_components.pjbi
                    AND arch = 'amd64'
                    AND platform LIKE "%metal%"
                    AND junit.modified_time BETWEEN DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 2 MONTH) AND CURRENT_DATETIME()
            
            UNION ALL    
            
            SELECT  *
            FROM `openshift-gce-devel.ci_analysis_us.junit_pr` junit_pr CROSS JOIN payload_components 
            WHERE   junit_pr.prowjob_build_id = payload_components.pjbi
                    AND arch = 'amd64'
                    AND platform LIKE "%metal%"
                    AND junit_pr.modified_time BETWEEN DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 2 MONTH) AND CURRENT_DATETIME()
        )
        # SELECT * FROM junit_all LIMIT 100
        SELECT  COUNT(DISTINCT junit_all.prowjob_build_id) AS unique_prowjobs, 
                network, 
                platform, 
                arch, 
                upgrade, 
                test_id, 
                ANY_VALUE(test_name) as test_name, 
                tag_source_location, 
                tag_commit_id, 
                MIN(first_release_date) as first_release_date, 
                MIN(created_at) as committed_at, 
                (COUNT(*)-SUM(success_val)-SUM(flake_count)) as fail_count, 
                SUM(success_val) as success_count, 
                SUM(flake_count) as flake_count
        FROM junit_all
        GROUP BY network, platform, arch, upgrade, test_id, tag_source_location,tag_commit_id
    '''

    df = client.query(QUERY).to_dataframe(create_bqstorage_client=True, progress_bar_type='tqdm')
    print(f'Found {len(df.index)} rows..')
    print('Sorting')
    df.sort_values(by=['committed_at'], ascending=False, inplace=True)
    print('Filtering')
    #df = df[df['unique_prowjobs'] > 10]

    col_success_acc: List[int] = list()
    col_failure_acc: List[int] = list()
    col_flake_acc: List[int] = list()

    print(f'Processing {len(df.index)} rows..')

    print(f'\nPhase 1...')
    counter = 0
    for row in df.itertuples():
        c = Coordinate(network=row.network, upgrade=row.upgrade, platform=row.platform, arch=row.arch, test_id=row.test_id)
        col_success_acc.append(success_sums[c].inc(row.success_count))
        col_failure_acc.append(failure_sums[c].inc(row.fail_count))
        col_flake_acc.append(flake_sums[c].inc(row.flake_count))
        counter += 1
        if counter % 100000 == 0:
            print(f'{counter // 100000}', end=' ')

    df['failure_acc'] = col_failure_acc
    df['success_acc'] = col_success_acc
    df['flake_acc'] = col_flake_acc

    col_p = list()
    col_failure_acc_before = list()
    col_success_acc_before = list()
    col_pass_rate = list()
    col_local_pass_rate = list()
    col_before_pass_rate = list()
    col_uids = list()

    regressed_test_uids = set()

    print(f'\nPhase 2...')
    counter = 0
    for row in df.itertuples():
        c = Coordinate(network=row.network, upgrade=row.upgrade, platform=row.platform, arch=row.arch, test_id=row.test_id)
        success_acc_before = success_sums[c].dec(row.success_count)
        failure_acc_before = failure_sums[c].dec(row.fail_count)
        before_pass_rate = 0
        if success_acc_before + failure_acc_before > 0:
            before_pass_rate = success_acc_before / (success_acc_before + failure_acc_before)

        col_local_pass_rate.append(row.success_count / (row.success_count + row.fail_count))

        col_before_pass_rate.append(before_pass_rate)
        col_failure_acc_before.append(failure_acc_before)
        col_success_acc_before.append(success_acc_before)
        pass_rate = row.success_acc / (row.failure_acc + row.success_acc)
        col_pass_rate.append(pass_rate)
        col_uids.append(c.uid())
        p = fast_fisher_cython.fisher_exact(row.failure_acc, row.success_acc,
                                            failure_acc_before, success_acc_before,
                                            alternative='greater')
        col_p.append(p)
        if before_pass_rate - pass_rate > 0.05 and row.unique_prowjobs > 10:
            regressed_test_uids.add(c.uid())
        counter += 1
        if counter % 100000 == 0:
            print(f'{counter // 100000}', end=' ')

    df['failure_before'] = col_failure_acc_before
    df['success_before'] = col_success_acc_before
    df['pass_rate'] = col_pass_rate
    df['before_pass_rate'] = col_before_pass_rate
    df['local_pass_rate'] = col_local_pass_rate
    df['uid'] = col_uids
    df['p'] = col_p
    # df = df[df['p'] < 0.05]

    pathlib.Path('tests').mkdir(parents=True, exist_ok=True)
    print(f'\nFound {len(regressed_test_uids)} regressed tests')
    for uid in regressed_test_uids:
        test_id_records = df.loc[df['uid'] == uid]
        test_id_records.to_csv(f'tests/{uid}.csv', index=False)

    # for uid in regressed_test_uids:
    #     test_id_records = df.loc[df['uid'] == uid]
    #     p = test_id_records.plot(x='tag_commit_id', y=['pass_rate'])
    #     plt.xticks(ticks=range(len(test_id_records['tag_commit_id'])),
    #                labels=test_id_records['tag_commit_id'],
    #                rotation='vertical')
    #     plt.tight_layout()
    #     plt.savefig(f'tests/{uid}.png', dpi=200)
    #     print(f'wrote {uid}.png')
    #     matplotlib.pyplot.close()


