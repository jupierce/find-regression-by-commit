import pandas
import pathlib
import matplotlib.pyplot as plt
import webbrowser

get_labels_last_release_date = ''

for csv_path in pathlib.Path('tests').glob('**/*bfe0d*.csv'):
    print(f'Loading {csv_path}')
    df = pandas.read_csv(str(csv_path))
    filename_base = str(csv_path)[0:-4]

    by_mod = df.groupby('modified_time').aggregate(
        {
            'modified_time': 'min',
            'test_name': 'min',
            'test_id': 'min',
            'fe10': 'mean',
            'fe20': 'mean',
            'fe30': 'mean',
        }
    )

    test_name = by_mod.iloc[0]['test_name']
    if 'disruption' in test_name:
        print(f'Skipping disruption test: {test_name}')
        continue

    print(by_mod.to_string())

    fig, ax = plt.subplots()
    tolerance = 5  # points

    def get_labels():
        global get_labels_last_release_date
        labels = []
        for row in by_mod.itertuples():
            labels.append(f'{row.modified_time}')
        return labels

    p = by_mod.plot(ax=ax, x='modified_time', y=['fe10', 'fe20', 'fe30'], figsize=(20, 10), picker=tolerance, color=['r', 'y', 'g'])

    ax.set_xticks(
        ticks=range(len(by_mod['modified_time'])),
        labels=get_labels(),
        rotation='vertical')
    ax.set_ylim([0, 3.05])

    plt.tight_layout()
    plt.grid(visible=True, axis='x')
    plt.title(test_name + ' | ' + by_mod.iloc[0]['test_id'])
    plt.savefig(f'{filename_base}.png')
    print(f'wrote {filename_base}.png')

    def on_pick(event):
        artist = event.artist
        x, y = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        print(f'Clicked ind {ind[0]}')
        mod_time = by_mod.iloc[ind[0]].modified_time
        print(f'Selected modified time: {mod_time}')
        for row in df.loc[df['modified_time'] == mod_time].itertuples():
            webbrowser.open(url=row.link, autoraise=True)

    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.show()
    plt.close()

