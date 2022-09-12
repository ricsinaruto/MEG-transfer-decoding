import os
import numpy as np
import mne
import osl
import yaml
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

#dataset_path = os.path.join('rich_data', 'subj2', 'sess4', 'oslpy', 'task_part1_4_raw_tsss_mc_epo.fif')
#epochs = mne.read_epochs(dataset_path, preload=False)

dataset_path = os.path.join('rich_data', 'RC', 'oslpy', 'task_part1_rc_tsss_mc_preproc_raw.fif')
raw = mne.io.read_raw_fif(dataset_path, preload=False)
events = mne.find_events(raw, min_duration=0.002)
epochs = mne.Epochs(raw,
                    events,
                    event_id=[2,3,4,5,6,7,8,11,12,13,14,15],
                    tmin=-0.1,
                    tmax=1.0,
                    baseline=(None, 0),
                    reject=None,
                    picks="meg",
                    decim=5,
                    preload=True)

cue_evoked = epochs['8'].average()

visual_channels = [1711, 1721, 1641, 1941, 1731, 1741, 1911, 1931, 1921, 2041, 2141, 2111, 2121, 2031, 2131, 2341, 2331, 2311, 2321, 2541, 2431, 2511, 2521, 2531,
                   '0421', '0631', '0711', '0431', '1821', '1841', '0741', '1831', '2011', '1041', '1111', '0721', '1141', '0731', '2211', '2241', '2231', '2021']
channel_list = ['MEG0121','MEG0122','MEG0123','MEG0131','MEG0132','MEG0133','MEG0221','MEG0222','MEG0223','MEG0231','MEG0232','MEG0233','MEG0311','MEG0312','MEG0313','MEG0321','MEG0322','MEG0323','MEG0331','MEG0332','MEG0333','MEG0341','MEG0342','MEG0343','MEG0411','MEG0412','MEG0413','MEG0421','MEG0422','MEG0423','MEG0431','MEG0432','MEG0433','MEG0441','MEG0442','MEG0443','MEG0621','MEG0622','MEG0623','MEG0631','MEG0632','MEG0633','MEG0711','MEG0712','MEG0713','MEG0721','MEG0722','MEG0723','MEG0731','MEG0732','MEG0733','MEG0741','MEG0742','MEG0743','MEG1041','MEG1042','MEG1043','MEG1131','MEG1132','MEG1133','MEG1221','MEG1222','MEG1223','MEG1231','MEG1232','MEG1233','MEG1311','MEG1312','MEG1313','MEG1321','MEG1322','MEG1323','MEG1331','MEG1332','MEG1333','MEG1341','MEG1342','MEG1343','MEG1411','MEG1412','MEG1413','MEG1521','MEG1522','MEG1523','MEG1531','MEG1532','MEG1533','MEG1541','MEG1542','MEG1543','MEG1611','MEG1612','MEG1613','MEG1621','MEG1622','MEG1623','MEG1631','MEG1632','MEG1633','MEG1641','MEG1642','MEG1643','MEG1721','MEG1722','MEG1723','MEG1731','MEG1732','MEG1733','MEG1811','MEG1812','MEG1813','MEG1821','MEG1822','MEG1823','MEG1831','MEG1832','MEG1833','MEG1841','MEG1842','MEG1843','MEG1911','MEG1912','MEG1913','MEG1921','MEG1922','MEG1923','MEG1931','MEG1932','MEG1933','MEG1941','MEG1942','MEG1943','MEG2011','MEG2012','MEG2013','MEG2021','MEG2022','MEG2023','MEG2031','MEG2032','MEG2033','MEG2041','MEG2042','MEG2043','MEG2111','MEG2112','MEG2113','MEG2121','MEG2122','MEG2123','MEG2211','MEG2212','MEG2213','MEG2221','MEG2222','MEG2223','MEG2231','MEG2232','MEG2233','MEG2241','MEG2242','MEG2243','MEG2311','MEG2312','MEG2313','MEG2321','MEG2322','MEG2323','MEG2341','MEG2342','MEG2343','MEG2411','MEG2412','MEG2413','MEG2421','MEG2422','MEG2423','MEG2431','MEG2432','MEG2433','MEG2441','MEG2442','MEG2443','MEG2521','MEG2522','MEG2523','MEG2641','MEG2642','MEG2643']

viz_channels = []
for i in visual_channels:
    chn = 'MEG' + str(i)
    viz_channels.append(chn)
    viz_channels.append(chn[:-1] + '2')
    viz_channels.append(chn[:-1] + '3')

channel_list = list(set(channel_list).difference(viz_channels))
print(channel_list)


#evk = cue_evoked.pick_channels(['MEG0111','MEG0112','MEG0113','MEG0121','MEG0122','MEG0123','MEG0131','MEG0132','MEG0133','MEG0141','MEG0142','MEG0143','MEG0211','MEG0212','MEG0213','MEG1411','MEG1412','MEG1413','MEG1511','MEG1512','MEG1513','MEG1641','MEG1642','MEG1643','MEG1721','MEG1722','MEG1723','MEG1731','MEG1732','MEG1733','MEG2111','MEG2112','MEG2113','MEG2131','MEG2132','MEG2133','MEG2311','MEG2312','MEG2313','MEG2321','MEG2322','MEG2323','MEG2331','MEG2332','MEG2333','MEG2411','MEG2412','MEG2413','MEG2431','MEG2432','MEG2433','MEG2441','MEG2442','MEG2443','MEG2511','MEG2512','MEG2513','MEG2521','MEG2522','MEG2523','MEG2531','MEG2532','MEG2533','MEG2541','MEG2542','MEG2543','MEG2631','MEG2632','MEG2633'])

cue_evoked.pick_channels(channel_list)
cue_evoked.plot_topo(merge_grads=True)
#mne.viz.plot_sensors(raw.info, block=True, show_names=True)

#selected_chn = epochs.plot_sensors('select', show=True, block=True)
#print(selected_chn[1])