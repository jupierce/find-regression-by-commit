import pandas
import pathlib
import matplotlib.pyplot as plt
import webbrowser

get_labels_last_release_date = ''

for csv_path in pathlib.Path('tests').glob('**/*cb921b4a3fa31e83daa90cc418bb1cbc*.csv'):
    print(f'Loading {csv_path}')
    df = pandas.read_csv(str(csv_path))
    filename_base = str(csv_path)[0:-4]

    df['include'] = 0
    interesting_indexes = df.index[(df['fe10'] > 0.95) | (df['fe20'] > 0.95) | (df['fe30'] > 0.95)].tolist()
    for interesting_index in interesting_indexes:
        for idx in range(interesting_index-5, interesting_index+5):
            if idx >= 0 and idx < len(df.index):
                df.at[idx, 'include'] = 1

    #print(df.to_string())
    # df = df[df['include'] == 1]
    # df = df.iloc[::-1]  # Reverse the order of the rows so that time moves left to right
    print(df.to_string())

    fig, ax = plt.subplots()
    tolerance = 5  # points

    def get_labels():
        global get_labels_last_release_date
        labels = []
        for row in df.itertuples():
            if row.tag_commit_id == '972c4d885c17f60d155b86e6a02f31b4c95fd155':
                labels.append(f'***{row.tag_commit_id}'[:7])
            else:
                labels.append(f'{row.tag_commit_id}'[:7])
        return labels

    p = df.plot(ax=ax, x='tag_commit_id', y=['fe10', 'fe20', 'fe30'], figsize=(20, 10), picker=tolerance, color=['r', 'y', 'g'])

    ax.set_xticks(
        ticks=range(len(df['tag_commit_id'])),
        labels=get_labels(),
        rotation='vertical')
    ax.set_ylim([0, 3.05])

    plt.tight_layout()
    plt.grid(visible=True, axis='x')
    plt.title(df.iloc[0]['test_id'])
    plt.savefig(f'{filename_base}.png')
    print(f'wrote {filename_base}.png')

    def on_pick(event):
        artist = event.artist
        x, y = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        print(f'Clicked ind {ind[0]}')
        url = df.iloc[ind[0]].tag_source_location + '/commit/' + df.iloc[ind[0]].tag_commit_id
        webbrowser.open(url=url, autoraise=True)

    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.show()
    plt.close()

