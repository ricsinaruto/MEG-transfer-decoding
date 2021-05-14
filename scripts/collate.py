import os

for s in range(200):
    subj = str(s) if s > 99 else '0' + str(s)
    if s < 10:
        subj = '00' + str(s)

    fold1 = 'sub-A2' + subj
    fold2 = 'sub-V1' + subj

    fold = ''
    if os.path.isdir(fold1):
        fold = fold1
    elif os.path.isdir(fold2):
        fold = fold2
    else:
        continue

    target_path = os.path.join('preprocessed_data', fold)
    if not os.path.isdir(target_path):
        os.mkdir(target_path)

    source = os.path.join(fold, 'preprocessed_data_new.mat')
    target = os.path.join(target_path, 'preprocessed_data_new.mat')
    os.system(f'cp {source} {target}')

    source = os.path.join(fold, 'good_samples_new.mat')
    target = os.path.join(target_path, 'good_samples_new.mat')
    os.system(f'cp {source} {target}')
