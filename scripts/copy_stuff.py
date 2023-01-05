import os
for i in range(10):
    inp = f'/well/woolrich/projects/disp_csaky/eeg/session{i+2}/preproc1_40hz_noica/inner_speech_long/sub{i}'
    out = f'/well/woolrich/projects/disp_csaky/eeg/preproc1_40hz_noica/inner_speech_long/sub{i}'

    # copy the dataset
    os.system(f'mv {inp} {out}')