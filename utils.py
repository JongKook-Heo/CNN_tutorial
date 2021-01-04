import torch

class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, num):
        self.sum += value
        self.count += num
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}:{self.avg}"

def accuracy(predictions, labels, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)

        _, topk_idx = predictions.topk(maxk, 1, True, True)

        topk_idx = topk_idx.t() #(maxk, batch_size)
        label_repeat = labels.view(1, -1).expand_as(topk_idx) #(maxk, batch_size)
        correct = topk_idx.eq(label_repeat)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1)

            correct_k = correct_k.float().sum(0, keepdim = True)
            res.append(correct_k.item())
        return res
class AccuracyMeter(object):
    def __init__(self, topk=(1,)):
        self.topk = topk
        self.meters = [AverageMeter("Acc@%d"%i) for i in topk]

    def reset(self):
        for meter in self.meters:
            meter.reset()

    def update(self, predictions, labels):
        acc_list = accuracy(predictions, labels, self.topk)
        num = labels.size(0)
        for meter, value in zip(self.meters, acc_list):
            meter.update(value, num)

    def __str__(self):
        return " ".join(str(meter) for meter in self.meters)

    def as_dict(self):
        return {meter.name: meter.avg for meter in self.meters}

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins) * 60)
    return elapsed_mins, elapsed_secs