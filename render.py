import pandas
import pathlib
import matplotlib.pyplot as plt
import webbrowser
from typing import Set
import time


def main():
    get_labels_last_release_date = ''

    for csv_path in pathlib.Path('tests').glob('**/*.csv'):
        print(f'Loading {csv_path}')
        df = pandas.read_csv(str(csv_path))
        filename_base = str(csv_path)[0:-4]
        png_path = pathlib.Path(f'{filename_base}.png')
        # if png_path.exists():
        #     print(f'Already rendered: {str(png_path)} - skipping')
        #     continue

        tmpdir = pathlib.Path('tmp')
        tmpdir.mkdir(parents=True, exist_ok=True)

        by_mod = df.groupby('release_name').aggregate(
            {
                'release_name': 'min',
                'test_name': 'min',
                'test_id': 'min',
                'fe10': 'max',
                'fe20': 'max',
                'fe30': 'max',
                'fe1000': 'max',
            }
        )

        test_name = by_mod.iloc[0]['test_name']
        if 'disruption' in test_name:
            print(f'Skipping disruption test: {test_name}')
            continue

        print(by_mod.to_string())

        fig, ax = plt.subplots(num=csv_path.name)
        fig.suptitle('nightly' if 'nightly' in csv_path.name else 'ci')

        tolerance = 5  # points

        def get_labels():
            global get_labels_last_release_date
            labels = []
            for row in by_mod.itertuples():
                labels.append(f'{row.release_name}')
            return labels

        p = by_mod.plot(ax=ax, x='release_name', y=['fe10', 'fe20', 'fe30', 'fe1000'], figsize=(20, 10), picker=tolerance, color=['r', 'orange', 'y', 'black'])

        ax.set_xticks(
            ticks=range(len(by_mod['release_name'])),
            labels=get_labels(),
            rotation='vertical')
        ax.set_ylim([0, 4.05])

        plt.tight_layout()
        plt.grid(visible=True, axis='x')
        plt.title(test_name + ' | ' + by_mod.iloc[0]['test_id'])
        plt.gcf().number = 't'
        plt.savefig(f'{str(png_path)}')
        print(f'wrote {str(png_path)}')

        last_click_x = -1
        last_click_y = -1

        def on_pick(event):
            nonlocal last_click_x, last_click_y
            artist = event.artist
            if last_click_y == event.mouseevent.y and last_click_x == event.mouseevent.x:
                # We've been invoked multiple times for the same click since multiple
                # lines were close by.
                return

            last_click_y = event.mouseevent.y
            last_click_x = event.mouseevent.x
            ind = event.ind
            print(f'Clicked ind {ind[0]}')
            release_name = by_mod.iloc[ind[0]].release_name
            print(f'Selected release_name time: {release_name}')

            html_path = tmpdir.joinpath(f'{release_name}.html')
            with html_path.open(mode='w+') as f:

                commit_detail_tables: list[str] = list()

                f.write(f'''
                <html>
                <head>
                    <title>{release_name}</title>
                    <style>
                        .styled-table {{
                            margin: 25px 0;
                            font-size: 0.9em;
                            font-family: sans-serif;
                            min-width: 400px;
                            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
                        }}      
                        .styled-table thead tr {{
                            background-color: #009879;
                            color: #ffffff;
                            text-align: left;
                        }}          
                        .styled-table th,
                        .styled-table td {{
                            padding: 12px 15px;
                        }}   
                        .styled-table tbody tr {{
                            border-bottom: 1px solid #dddddd;
                        }}
                        .styled-table tbody tr:nth-of-type(even) {{
                            background-color: #f3f3f3;
                        }}
                        .styled-table tbody tr:last-of-type {{
                            border-bottom: 2px solid #009879;
                        }}
                        .styled-table tbody tr.active-row {{
                            font-weight: bold;
                            color: #009879;
                        }}                                                                                                 
                    </style>
                </head>
                <body>
                    <h2>{test_name}</h2>
                    <h2>{csv_path.name}</h2>
                    <h2>{release_name}</h2>
                    <br>
                    <table class="styled-table">
                        <tr> 
                            <th>fe10</th>

                            <th>fe20</th>

                            <th>fe30</th>

                            <th>fe1000</th>

                            <th>commit</th>
                            
                            <th>source location</th>
                        </tr>
''')

                carried_commits = df.loc[df['release_name'] == release_name]
                for row in carried_commits.sort_values([
                    'fe10',
                    'fe20',
                    'fe30',
                    'fe1000'
                ], ascending=False).itertuples():

                    link = row.source_location + '/commit/' + row.tag_commit_id
                    if link.startswith('https://github.com/'):
                        org, repo, _, commit = link.split('/')[3:]
                    else:
                        org = link
                        repo = ''
                        commit = row.tag_commit_id
                    f.write(f'''
                        <tr>
                            <td>{row.fe10/4:.3f}</td>

                            <td>{row.fe20/3:.3f}</td>

                            <td>{row.fe30/2:.3f}</td>

                            <td>{row.fe1000:.3f}</td>
                            
                            <td><a href="#{commit}">{commit}</a></td>
                            
                            <td>{org}/{repo}</td> 
                        </tr>
''')
                    def success_rate(success: int, failure: int) -> str:
                        pass_rate = 1.0
                        if failure + success != 0:
                            pass_rate = success / (success + failure)
                        pass_rate *= 100.0
                        return f'{pass_rate:.1f}%'

                    commit_detail = f'''
                        <br>
                        <a id="{commit}"/>
                        <b>Source:</b> <a href="{link}">{link}</a>
                        <table class="styled-table">
                            <tr> 
                                <th>type</th>
                                <th colspan="3">fe10</th>
                                <th colspan="3">fe20</th>
                                <th colspan="3">fe30</th>
                                <th colspan="3">fe1000</th>
                            </tr>
                            <tr>
                                <th>Before Commit</th>
                                <td style='color:green;'>{row.b_s10}</td>
                                <td style='color:red;'>{row.b_f10}</td>
                                <td>{success_rate(row.b_s10, row.b_f10)}</td>
                                
                                <td style='color:green;'>{row.b_s20}</td>
                                <td style='color:red;'>{row.b_f20}</td>
                                <td>{success_rate(row.b_s20, row.b_f20)}</td>

                                <td style='color:green;'>{row.b_s30}</td>
                                <td style='color:red;'>{row.b_f30}</td>
                                <td>{success_rate(row.b_s30, row.b_f30)}</td>
                                
                                <td style='color:green;'>{row.b_s1000}</td>
                                <td style='color:red;'>{row.b_f1000}</td>
                                <td>{success_rate(row.b_s1000, row.b_f1000)}</td>
                            <tr>
                            <tr>
                                <th>After Commit</th>
                                <td style='color:green;'>{row.a_s10}</td>
                                <td style='color:red;'>{row.a_f10}</td>
                                <td>{success_rate(row.a_s10, row.a_f10)}</td>

                                <td style='color:green;'>{row.a_s20}</td>
                                <td style='color:red;'>{row.a_f20}</td>
                                <td>{success_rate(row.a_s20, row.a_f20)}</td>

                                <td style='color:green;'>{row.a_s30}</td>
                                <td style='color:red;'>{row.a_f30}</td>
                                <td>{success_rate(row.a_s30, row.a_f30)}</td>

                                <td style='color:green;'>{row.a_s1000}</td>
                                <td style='color:red;'>{row.a_f1000}</td>
                                <td>{success_rate(row.a_s1000, row.a_f1000)}</td>
                            <tr>
                            <tr>
                                <th>Regression Probability</th>
                                <td colspan="3" style="text-align: center;">{row.fe10/4*100:.2f}%</td>
                                <td colspan="3" style="text-align: center;">{row.fe20/3*100:.2f}%</td>
                                <td colspan="3" style="text-align: center;">{row.fe30/2*100:.2f}%</td>
                                <td colspan="3" style="text-align: center;">{row.fe1000*100:.2f}%</td>
                            </th>
                        </table>
'''
                    commit_detail_tables.append(commit_detail)

                f.write(f'''
                    </table>
                    <br>
''')

                for commit_detail in commit_detail_tables:
                    f.write('<br>\n')
                    f.write(commit_detail)
                    f.write('<br>\n')

                f.write(f'''
                </body>
                </html>
''')
            webbrowser.open(url=html_path.absolute().as_uri(), autoraise=True)

        fig.canvas.callbacks.connect('pick_event', on_pick)
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()