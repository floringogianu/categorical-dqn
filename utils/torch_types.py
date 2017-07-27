import torch


class TorchTypes(object):

    def __init__(self, cuda=False):
        self.set_cuda(cuda)

    def set_cuda(self, use_cuda):
        if use_cuda:
            self.ByteTensor = torch.cuda.ByteTensor
            self.IntTensor = torch.cuda.IntTensor
            self.LongTensor = torch.cuda.LongTensor
            self.FloatTensor = torch.cuda.FloatTensor
            self.DoubleTensor = torch.cuda.DoubleTensor
        else:
            self.FloatTensor = torch.FloatTensor
            self.DoubleTensor = torch.DoubleTensor
            self.LongTensor = torch.LongTensor
            self.IntTensor = torch.IntTensor
            self.ByteTensor = torch.ByteTensor
