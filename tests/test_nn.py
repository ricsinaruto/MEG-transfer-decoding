import os
import ast

# load loss files
directory = os.path.join('results_test', 'neural_network', 'subj0')
names = ['val', 'test']
values = {'valloss/Validation loss: ': 0.29926426087816554, 'valloss/valcriterion/Validation accuracy: ': 0.5974576274553934, 'valloss/saveloss/none': 0.4025423725446065}
keys = list(values.keys())
for name in names:
    path = os.path.join(directory, name + '_loss.txt')
    with open(path, 'r') as f:
        lines = f.readlines()

    d = lines[0].strip()
    # use eval to convert string to dict
    d = ast.literal_eval(d)

    # assert that loss is the same as val to 4 decimal places
    for k in keys:
        assert abs(float(d[k])-float(values[k])) < 0.005

# check that model.pt exists
for n in ['_end', '_epoch', '_init', '']:
    path = os.path.join(directory, 'model' + n + '.pt')
    assert os.path.exists(path)

# check that losses.svg exists
path = os.path.join(directory, 'losses.svg')
assert os.path.exists(path)

# assert that args_saved.py is same as args.py
with open('args.py', 'r') as f:
    args = f.readlines()
with open(os.path.join(directory, 'args_saved.py'), 'r') as f:
    args_saved = f.readlines()

assert args == args_saved
