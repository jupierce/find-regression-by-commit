import pandas
import pathlib
import matplotlib.pyplot as plt
import webbrowser
from typing import Set
import time


def main():
    get_labels_last_release_date = ''

    for csv_path in pathlib.Path('tests').glob('**/*.csv'):  #'**/*bfe0d*.csv'):
        print(f'Loading {csv_path}')
        df = pandas.read_csv(str(csv_path))
        filename_base = str(csv_path)[0:-4]

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
        plt.savefig(f'{filename_base}.png')
        print(f'wrote {filename_base}.png')

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
                f.write(f'''
                <html>
                <head>
                    <title>{release_name}</title>
                </head>
                <body>
                    <h2>{test_name}</h2>
                    <h2>{csv_path.name}</h2>
                    <h2>{release_name}</h2>
                    <br>
                    <table>
                        <tr>
                            <th>fe10</th>
                            <th>fe20</th>
                            <th>fe30</th>
                            <th>fe1000</th>
                            <th>commit</th>
                        </tr>
''')

                carried_commits = df.loc[df['release_name'] == release_name]
                for row in carried_commits.sort_values(['fe10', 'fe20', 'fe30', 'fe1000'], ascending=False).itertuples():
                    f.write(f'''
                        <tr>
                            <td>{row.fe10}</td>
                            <td>{row.fe20}</td>
                            <td>{row.fe30}</td>
                            <td>{row.fe1000}</td>
                            <td><a href="{row.link}">{row.link}</a></td>
                        </tr>
''')
                f.write(f'''
                    </table>
                    <br>
                </body>
                </html>
''')
            webbrowser.open(url=html_path.absolute().as_uri(), autoraise=True)

        fig.canvas.callbacks.connect('pick_event', on_pick)
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()