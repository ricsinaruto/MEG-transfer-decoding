import os

# load loss files
directory = os.path.join('results_test', 'lda_nn', 'subj0', 'sliding_windows')
values = os.path.join('tests', 'results', 'lda_nn', 'subj0', 'sliding_windows')
names = ['train', 'val', 'test']
for name in names:
    path = os.path.join(directory, name + '_loss.txt')

    losses = []
    with open(path, 'r') as f:
        for line in f:
            losses.append(float(line.strip()))

    # load saved values
    true_losses = []
    path = os.path.join(values, name + '_loss.txt')
    with open(path, 'r') as f:
        for line in f:
            true_losses.append(float(line.strip()))

    # assert that losses are the same as true_losses to 4 decimal places
    for loss, val in zip(losses, true_losses):
        assert round(loss, 4) == round(val, 4)

# check that model.pt5 ... model.pt105 exists
for i in range(5, 105):
    path = os.path.join(directory, 'model.pt' + str(i))
    assert os.path.exists(path)

# assert that args_saved.py is same as args.py
with open('args.py', 'r') as f:
    args = f.readlines()
with open(os.path.join(directory, 'args_saved.py'), 'r') as f:
    args_saved = f.readlines()

assert args == args_saved
