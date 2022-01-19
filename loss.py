class Loss:
    '''
    Class for accumulating and printing losses.
    Losses are handled by a dict containing different metrics.
    '''
    def __init__(self):
        self.dict = {}

    def append(self, x):
        for key in x:
            if self.dict.get(key, 'no') == 'no':
                self.dict[key] = [x[key].item()]
            else:
                self.dict[key].append(x[key].item())

    def print(self, split, exception='saveloss'):
        msg = ''
        for k, v in self.dict.items():
            self.dict[k] = sum(v)/len(v)

            if split in k and exception not in k:
                msg += k.split('/')[-1] + ': ' + str(self.dict[k]) + '\t'

        print(msg, flush=True)

        losses = dict([(k, self.dict[k]) for k in self.dict if split in k])
        self.dict = {}
        return losses
