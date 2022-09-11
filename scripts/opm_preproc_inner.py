events = pd.read_csv(path, sep='\t')

event_c = np.array([events['value'][i] for i in range(len(events))])
event_t = np.array([events['sample'][i] for i in range(len(events))])

count1 = 0
count2 = 0
new_events = []
for ind, (et, ec) in enumerate(zip(event_t[1:-1], event_c[1:-1])):
    think_trial = False
    i = ind + 1
    if ec < 7 and ec > 1 and (event_t[i+1] - et) > 5 and (et - event_t[i-1]) > 5:
        count1 += 1
        if event_c[i+2] == 8:
            new_events.append(np.array([event_t[i+2], 0, ec]))
        else:
            print('error1')
        if event_c[i+3] == 8:
            new_events.append(np.array([event_t[i+3], 0, ec]))
        else:
            print('error2')
        if event_c[i+4] == 8:
            new_events.append(np.array([event_t[i+4], 0, ec]))
        else:
            print('error3')
        if event_c[i+5] == 8:
            new_events.append(np.array([event_t[i+5], 0, ec]))
        else:
            print('error4')

    elif (event_t[i+1] - et) < 5 and ec > 1:
        ec += event_c[i+1]
        think_trial = True

    if think_trial:
        count2 += 1
        split_events = event_c[i-18:i-4]
        #print(split_events)
        tind = np.nonzero(split_events == 7)[0][-1]

        tind += i-18

        if event_c[tind+2] == 8:
            new_events.append(np.array([event_t[tind+2], 0, ec-9]))
        else:
            print('erorr5')
            print(event_c[i-20:i])
        if event_c[tind+3] == 8:
            new_events.append(np.array([event_t[tind+3], 0, ec-9]))
        else:
            print('erorr6')
            print(event_c[i-20:i])
        if event_c[tind+4] == 8:
            new_events.append(np.array([event_t[tind+4], 0, ec-9]))
        else:
            print('erorr7')
        if event_c[tind+5] == 8:
            new_events.append(np.array([event_t[tind+5], 0, ec-9]))
        else:
            print('erorr8')

print(count1)
print(count2)

new_events = np.array(new_events)