import pandas

df = pandas.read_csv('output.csv')
df = df[df['success_acc'] > 20]
df = df[df['failure_acc'] > 20]
df = df[df['tag_commit_id'] != '']

groups = df.groupby('test_id')
for name, group in groups:
    # min = group[group.p == group.p.min()]
    # min.to_csv(f'tests/{name}.csv')
    group.to_csv(f'tests/{name}.csv')