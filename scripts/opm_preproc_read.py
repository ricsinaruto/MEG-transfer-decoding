# reading trials
events = pd.read_csv('rich_data/opm_rich/Task/20220908_115229_events.tsv', sep='\t')

event_c = np.array([events['value'][i] for i in range(len(events))])
event_t = np.array([events['sample'][i] for i in range(len(events))])

new_events = []
for ind, (et, ec) in enumerate(zip(event_t[1:-1], event_c[1:-1])):
    i = ind + 1
    if ec < 7 and ec > 1 and (event_t[i+1] - et) > 5 and (et - event_t[i-1]) > 5:
        new_events.append(np.array([et, 0, ec]))