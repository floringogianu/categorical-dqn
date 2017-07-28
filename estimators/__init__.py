# Bitdefender, 2107
from estimators.atari_net import AtariNet
from estimators.catch_net import CatchNet

ESTIMATORS = {
    "atari": AtariNet,
    "catch": CatchNet
}


def get_estimator(name, in_ch, hist_len, action_no, hidden_size=128):
    return ESTIMATORS[name](in_ch, hist_len, action_no, hidden_size)
