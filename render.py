import pandas
import pathlib
import matplotlib.pyplot as plt
import webbrowser

for csv_path in pathlib.Path('tests').glob('*metal*.csv'):
    df = pandas.read_csv(str(csv_path))
    str_name = str(csv_path)[0:-4]

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    tolerance = 5  # points

    df['local_pass_rate'] = df['success_count'] / (df['success_count'] + df['fail_count'])

    def get_labels():
        labels = []
        for row in df.itertuples():
            labels.append(row.first_release_date + '-' + row.tag_commit_id[:7])
        return labels

    p = df.plot(ax=ax, x='tag_commit_id', y=['pass_rate', 'local_pass_rate'], figsize=(20, 10), picker=tolerance, color=['g', 'y'])
    p2 = df.plot(ax=ax2, x='tag_commit_id', y=['unique_prowjobs'], figsize=(20, 10), picker=tolerance)
    ax.set_xticks(
        ticks=range(len(df['tag_commit_id'])),
        labels=get_labels(),
        rotation='vertical')
    ax.set_ylim([0, 1])
    #ax2.set_ylim([0, 1])
    plt.tight_layout()
    plt.grid(visible=True, axis='x')
    plt.title(str_name + ' - ' + df['test_name'][0])
    plt.savefig(f'{str_name}.png')
    print(f'wrote {str_name}.png')

    def on_pick(event):
        artist = event.artist
        x, y = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        url = df.iloc[ind[0]].tag_source_location + '/commit/' + df.iloc[ind[0]].tag_commit_id
        webbrowser.open(url=url, autoraise=True)

    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.show()
    plt.close()

