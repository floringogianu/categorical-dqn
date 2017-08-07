import torch


class TorchTypes(object):

    def __init__(self, cuda=False):
        self.set_cuda(cuda)

    def set_cuda(self, use_cuda):
        if use_cuda:
            self.FT = torch.cuda.FloatTensor
            self.LT = torch.cuda.LongTensor
            self.BT = torch.cuda.ByteTensor
            self.IT = torch.cuda.IntTensor
            self.DT = torch.cuda.DoubleTensor
        else:
            self.FT = torch.FloatTensor
            self.LT = torch.LongTensor
            self.BT = torch.ByteTensor
            self.IT = torch.IntTensor
            self.DT = torch.DoubleTensor
