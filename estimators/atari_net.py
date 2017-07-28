""" Neural Network architecture for Atari games.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariNet(nn.Module):
    def __init__(self, input_channels, hist_len, out_size, hidden_size=256):
        super(AtariNet, self).__init__()
        self.input_channels = input_channels
        self.hist_len = hist_len
        self.input_depth = input_depth = hist_len * input_channels
        if type(out_size) is tuple:
            self.is_categorical = True
            self.action_no, self.atoms_no = out_size
            self.out_size = self.action_no * self.atoms_no
        else:
            self.is_categorical = False
            self.out_size = out_size
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(input_depth, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lin1 = nn.Linear(64 * 7 * 7, self.hidden_size)
        self.head = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        out = self.head(x.view(x.size(0), -1))
        if self.is_categorical:
            splits = out.chunk(self.action_no, 1)
            return torch.stack(list(map(lambda s: F.softmax(s), splits)), 1)
        else:
            return out

    def get_attributes(self):
        return (self.input_channels, self.hist_len, self.action_no,
                self.hidden_size)
