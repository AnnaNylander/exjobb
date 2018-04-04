
class ResultMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.epoch = 0
        self.batch = 0
        self.step = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, epoch, batch, step, val, n=1):
        self.values.append([step, val])
        self.epoch = epoch
        self.batch = batch
        self.step = step
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def serialize(self):
        dic = {}
        dic['values'] = self.values
        dic['epoch'] = self.epoch
        dic['batch'] = self.batch
        dic['step'] = self.step
        dic['val'] = self.val
        dic['avg'] = self.avg
        dic['sum'] = self.sum
        dic['count'] = self.count
        return dic

    def deserialize(self, dic):
        self.values = dic['values']
        self.epoch = dic['epoch']
        self.batch = dic['batch']
        self.step = dic['step']
        self.val = dic['val']
        self.avg = dic['avg']
        self.sum = dic['sum']
        self.count = dic['count']
