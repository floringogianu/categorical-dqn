from collections import namedtuple


""" A container for transitions. """
Transition = namedtuple('Transition',
                        ('state', 'action', 'state_', 'reward', 'done'))
