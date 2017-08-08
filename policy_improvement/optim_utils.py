import itertools
import torch.optim as optim


def float_range(start, end, step):
    x = start
    if step > 0:
        while x < end:
            yield x
            x += step
    else:
        while x > end:
            yield x
            x += step


def lr_schedule(start, end, steps_no):
    start, end, steps_no = float(start), float(end), float(steps_no)
    step = (end - start) / (steps_no - 1.)
    schedules = [float_range(start, end, step), itertools.repeat(end)]
    return itertools.chain(*schedules)


def optim_factory(weights, cmdl):
    if cmdl.optim == "Adam":
        return optim.Adam(weights, lr=cmdl.lr, eps=cmdl.eps)
    elif cmdl.optim == "RMSprop":
        return optim.RMSprop(weights, lr=cmdl.lr, eps=cmdl.eps,
                             alpha=cmdl.alpha)
