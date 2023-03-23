import os

# load loss files
directory = os.path.join('results_test', 'lda_pca', 'subj0')
names = ['train', 'val', 'test']
values = [0.9978813559322034, 0.2937853107344633, 0.2937853107344633]
for name, val in zip(names, values):
    path = os.path.join(directory, name + '_loss.txt')
    with open(path, 'r') as f:
        lines = f.readlines()

    loss = float(lines[0].strip())

    # assert that loss is the same as val to 4 decimal places
    assert round(loss, 4) == round(val, 4)

# check that model.pt25 exists
path = os.path.join(directory, 'model.pt25')
assert os.path.exists(path)

# assert that args_saved.py is same as args.py
with open('args.py', 'r') as f:
    args = f.readlines()
with open(os.path.join(directory, 'args_saved.py'), 'r') as f:
    args_saved = f.readlines()

assert args == args_saved
