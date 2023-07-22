#!/usr/bin/env python3

from collections import defaultdict
from typing import NamedTuple, List

import matplotlib.pyplot
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


client = bigquery.Client(project='openshift-gce-devel')


class Coordinate(NamedTuple):
    network: str
    upgrade: str
    platform: str
    arch: str
    test_id: str
    source_location: str

    def uid(self):
        return f'{self.network}-{self.upgrade}-{self.arch}-{self.platform}-{self.test_id}'


success_sums = defaultdict(WrappedInteger)
failure_sums = defaultdict(WrappedInteger)
flake_sums = defaultdict(WrappedInteger)


class ResultSum:
    def __init__(self, success_count=0, failure_count=0, flake_count=0):
        self.success_count = success_count
        self.failure_count = failure_count
        self.flake_count = flake_count

    def pass_rate(self) -> float:
        if self.success_count > 0 or self.failure_count > 0:
            return self.success_count / (self.success_count + self.failure_count)
        return 0.0

    def fishers_exact_regressed(self, result_sum2) -> float:
        # If at least ten tests have run including this commit and
        # if we have regressed at least 10% relative to the result we are comparing to.
        if self.success_count + self.failure_count > 10 and result_sum2.pass_rate() - self.pass_rate() > 0.10:
            return 1 - fast_fisher_cython.fisher_exact(
                self.failure_count, self.success_count,
                result_sum2.failure_count, result_sum2.success_count,
                alternative='greater')

        return 0.0

    def __str__(self):
        return f'[s={self.success_count} f={self.failure_count} r={int(self.pass_rate() * 100)}%]'

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

    repos = pathlib.Path('payload-repos.txt').read_text().strip().splitlines()
    repos_csv = ','.join([f'"{repo}"' for repo in repos])  # "openshift/repo","openshift/repo2"

    # LIMITING TO AMD64 for now! See WHERE CLAUSE
    QUERY = f'''
        WITH junit_all AS (
            WITH payload_components AS(               
                # Find all prowjobs which have run against a 4.14 payload commit
                # in the last two months. 
                SELECT  prowjob_build_id as pjbi, 
                        tag_source_location, 
                        tag_commit_id, 
                        ANY_VALUE(release_name), 
                        MIN(release_created) as first_release_date, 
                FROM openshift-gce-devel.ci_analysis_us.job_releases jr
                WHERE   release_created BETWEEN DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 2 MONTH) AND CURRENT_DATETIME()
                        AND release_name LIKE "4.14.%"   
                        AND tag_source_location LIKE "%cluster-storage-operator%"             
                GROUP BY prowjob_build_id, tag_source_location, tag_commit_id
            )
            
            # Find all junit tests run in non-PR triggered tests which ran as part of those prowjobs over the last two months
            SELECT  *
            FROM `openshift-gce-devel.ci_analysis_us.junit` junit CROSS JOIN payload_components 
            WHERE   junit.prowjob_build_id = payload_components.pjbi
                    AND arch = 'amd64'
                    AND platform LIKE "%metal%"
                    AND junit.modified_time BETWEEN DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 2 MONTH) AND CURRENT_DATETIME()
            
            UNION ALL    
            
            # Find all junit tests run in PR triggered tests which ran as part of those prowjobs over the last two months
            SELECT  *
            FROM `openshift-gce-devel.ci_analysis_us.junit_pr` junit_pr CROSS JOIN payload_components 
            WHERE   junit_pr.prowjob_build_id = payload_components.pjbi
                    AND arch = 'amd64'
                    AND platform LIKE "%metal%"
                    AND junit_pr.modified_time BETWEEN DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 2 MONTH) AND CURRENT_DATETIME()
        )
        
        SELECT  COUNT(DISTINCT junit_all.prowjob_build_id) AS unique_prowjobs, 
                ANY_VALUE(network), 
                ANY_VALUE(platform), 
                ANY_VALUE(arch), 
                ANY_VALUE(upgrade), 
                test_id, 
                ANY_VALUE(test_name) as test_name, 
                tag_source_location, 
                tag_commit_id, 
                MIN(first_release_date) as first_release_date, 
                (COUNT(*)-SUM(success_val)-SUM(flake_count)) as fail_count, 
                SUM(success_val) as success_count, 
                SUM(flake_count) as flake_count
        FROM junit_all
        GROUP BY    #network, 
                    #platform, 
                    #arch, 
                    #upgrade,    
                    test_id, 
                    tag_source_location, 
                    tag_commit_id
        # TODO: We want the records ordered by commit date. This is an imperfect approximation
        # because not all commits are represented by the source_commit in a CI payload and
        # nightly payload. The payload constructed for PR testing seems to have branch names
        # instead of commit shas. 
        # This means commit A could merge followed by commit B. Because of debound in the
        # release controller, it could decide to test B for the CI payload, but the ART
        # builds might update nightlies with commit A. In this case, our simple sort 
        # here would come to conclude that B precedes A, which is false. In practice,
        # this should be rare.
        ORDER BY first_release_date DESC   
    '''

    df = client.query(QUERY).to_dataframe(create_bqstorage_client=True, progress_bar_type='tqdm')
    print(f'Found {len(df.index)} rows..')

    lookahead_10_row: ResultSum = ResultSum()
    lookbehind_10_row: ResultSum = ResultSum()

    lookahead_20_row: ResultSum = ResultSum()
    lookbehind_20_row: ResultSum = ResultSum()

    lookahead_30_row: ResultSum = ResultSum()
    lookbehind_30_row: ResultSum = ResultSum()

    # Initialize columns we will be populating during the scan
    df['lookahead10'] = ResultSum()
    df['lookbehind10'] = ResultSum()
    df['fe10'] = 0.0

    df['lookahead20'] = ResultSum()
    df['lookbehind20'] = ResultSum()
    df['fe20'] = 0.0

    df['lookahead30'] = ResultSum()
    df['lookbehind30'] = ResultSum()
    df['fe30'] = 0.0

    print(f'Processing {len(df.index)} rows..')

    grouped = df.groupby(['tag_source_location', 'test_id'])
    for name, group in grouped:
        look_success: List[int] = list()
        look_failure: List[int] = list()
        look_flake: List[int] = list()

        for idx, row in group.iterrows():
            look_success.append(row['success_count'])
            look_failure.append(row['fail_count'])
            look_flake.append(row['flake_count'])

            for look_size in (10, 20, 30):
                group.at[idx, f'lookahead{look_size}'] = ResultSum(
                    success_count=sum(look_success[-1 * look_size:]),
                    failure_count=sum(look_failure[-1 * look_size:]),
                    flake_count=sum(look_flake[-1 * look_size:]),
                )

        for idx, row in group.iterrows():
            look_success.pop(0)
            look_failure.pop(0)
            look_flake.pop(0)
            for look_size in (10, 20, 30):
                group.at[idx, f'lookbehind{look_size}'] = ResultSum(
                    success_count=sum(look_success[0:look_size]),
                    failure_count=sum(look_failure[0:look_size]),
                    flake_count=sum(look_flake[0:look_size]),
                )

        output_to_csv = False
        for idx, row in group.iterrows():
            for look_size in (10, 20, 30):
                fe = row[f'lookahead{look_size}'].fishers_exact_regressed(row[f'lookbehind{look_size}'])
                if fe > 0.95:
                    output_to_csv = True
                group.at[idx, f'fe{look_size}'] = fe

        if output_to_csv:
            tag_source_location = group.iloc[0]['tag_source_location']
            test_id = group.iloc[0]['test_id']
            org, repo = tag_source_location.split('/')[-2:]
            orgdir = pathlib.Path(f'assessments/{org}/{repo}')
            orgdir.mkdir(exist_ok=True, parents=True)
            repo_out = orgdir.joinpath(f'{test_id}.csv')
            group.to_csv(str(repo_out))

    exit(0)


    col_p = list()
    col_failure_acc_before = list()
    col_success_acc_before = list()
    col_pass_rate = list()
    col_highlight = list()
    col_local_pass_rate = list()
    col_before_pass_rate = list()
    col_uids = list()

    regressed_test_uids = set()

    print(f'\nPhase 2...')
    counter = 0
    for row in df.itertuples():
        c = Coordinate(network=row.network, upgrade=row.upgrade, platform=row.platform, arch=row.arch, test_id=row.test_id, source_location=row.tag_source_location)
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
            col_highlight.append('HERE')
        else:
            col_highlight.append('_')
        counter += 1
        if counter % 100000 == 0:
            print(f'{counter // 100000}', end=' ')

    df['failure_before'] = col_failure_acc_before
    df['success_before'] = col_success_acc_before
    df['pass_rate'] = col_pass_rate
    df['before_pass_rate'] = col_before_pass_rate
    df['local_pass_rate'] = col_local_pass_rate
    df['hl'] = col_highlight
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


