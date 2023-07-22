import pandas
import pathlib
import matplotlib.pyplot as plt
import webbrowser

get_labels_last_release_date = ''

for csv_path in pathlib.Path('assessments').glob('**/*.csv'):
    df = pandas.read_csv(str(csv_path))
    df = df.iloc[::-1]  # Reverse the order of the rows so that time moves left to right
    filename_base = str(csv_path)[0:-4]

    fig, ax = plt.subplots()
    tolerance = 5  # points

    def get_labels():
        global get_labels_last_release_date
        labels = []
        for row in df.itertuples():
            labels.append(row.tag_commit_id[:7])
        return labels

    p = df.plot(ax=ax, x='tag_commit_id', y=['fe10', 'fe20', 'fe30'], figsize=(20, 10), picker=tolerance, color=['r', 'y', 'g'])

    ax.set_xticks(
        ticks=range(len(df['tag_commit_id'])),
        labels=get_labels(),
        rotation='vertical')
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.grid(visible=True, axis='x')
    plt.title(df.loc[0]['tag_source_location'] + ' - ' + df['test_name'][0])
    plt.savefig(f'{filename_base}.png')
    print(f'wrote {filename_base}.png')

    def on_pick(event):
        artist = event.artist
        x, y = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        url = df.iloc[ind[0]].tag_source_location + '/commit/' + df.iloc[ind[0]].tag_commit_id
        webbrowser.open(url=url, autoraise=True)

    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.show()
    plt.close()

