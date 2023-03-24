import os

# load loss files
directory = os.path.join('results_test', 'lda_nn', 'subj0')
val = 0.9476676807185281

path = os.path.join(directory, 'pairwise.txt')
with open(path, 'r') as f:
    lines = f.readlines()

loss = float(lines[0].strip())

# assert that loss is the same as val to 4 decimal places
assert round(loss, 4) == round(val, 4)
