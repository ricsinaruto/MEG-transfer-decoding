class Loss:
    '''
    Class for accumulating and printing losses.
    '''
    def __init__(self):
        self.list = []
        self.list2 = []

    def append(self, x, x2=None):
        self.list.append(x.item())
        if x2 is not None:
            self.list2.append(x2.item())

    def print(self, message):
        loss = sum(self.list)/len(self.list)
        if self.list2:
            loss2 = sum(self.list2)/len(self.list2)
            print(message, loss, ' ', loss2)
        else:
            print(message, loss)

        self.list = []
        self.list2 = []
        return loss
