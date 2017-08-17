# Tudor Berariu @ Bitdefender, 2017

import torch


class BilinearResizer(object):

    def __init__(self, cuda=True, orig_size=None, new_size=None):
        self.cuda = cuda
        if orig_size is not None and new_size is not None:
            self._compute_idxs(orig_size, new_size)
        else:
            self.orig_size = None
            self.new_size = None

    def _compute_idxs(self, orig_size, new_size):
        assert len(orig_size) == 3 and len(new_size) == 3 and \
               orig_size[0] == new_size[0]

        d, o_h, o_w = orig_size
        _, n_h, n_w = new_size

        cuda = self.cuda

        r_idx = torch.linspace(0, o_h - 1, n_h).cuda()
        i_up = torch.floor(r_idx)
        i_down = i_up + 1.
        f_down = (r_idx - i_up).view(1, n_h, 1)
        f_up = (i_down - r_idx).view(1, n_h, 1)

        self.i_up = i_up.long()
        self.i_down = i_down.clamp(0, o_h - 1).long()
        self.f_down = f_down
        self.f_up = f_up

        if cuda:
            self.i_up, self.i_down = self.i_up.cuda(), self.i_down.cuda()
            self.f_up, self.f_down = self.f_up.cuda(), self.f_down.cuda()

        r_idx = torch.linspace(0, o_w - 1, n_w).cuda()
        i_left = torch.floor(r_idx)
        i_right = i_left + 1.
        f_left = (r_idx - i_right).view(1, 1, n_w)
        f_right = (i_left - r_idx).view(1, 1, n_w)

        self.i_left = i_left.long()
        self.i_right = i_right.clamp(0, o_w - 1).long()
        self.f_left = f_left
        self.f_right = f_right

        if cuda:
            self.i_left, self.i_right = self.i_left.cuda(), self.i_right.cuda()
            self.f_left, self.f_right = self.f_left.cuda(), self.f_right.cuda()

        self.orig_size = orig_size
        self.new_size = new_size

    def __call__(self, orig_img, new_size):
        orig_size = orig_img.size()
        if not(self.orig_size == orig_size and self.new_size == new_size):
            self._compute_idxs(orig_size, new_size)

        d, o_h, o_w = orig_size
        _, n_h, n_w = new_size

        z = orig_img.index_select(1, self.i_up) *\
            self.f_up.expand(d, n_h, o_w) +\
            orig_img.index_select(1, self.i_down) *\
            self.f_down.expand(d, n_h, o_w)

        y = z.index_select(2, self.i_left) * self.f_left.expand(d, n_h, n_w) +\
            z.index_select(2, self.i_right) * self.f_right.expand(d, n_h, n_w)

        return y


class NNResizer(object):

    def __init__(self, cuda=False, orig_size=None, new_size=None):
        self.cuda = cuda
        if orig_size is not None and new_size is not None:
            self._compute_idxs(orig_size, new_size)
        else:
            self.orig_size = None
            self.new_size = None

    def _compute_idxs(self, orig_size, new_size):
        assert len(orig_size) == 3 and len(new_size) == 3 and \
               orig_size[0] == new_size[0]

        d, o_h, o_w = orig_size
        _, n_h, n_w = new_size

        self.i_v = torch.linspace(0, o_h - 1, n_h).round().long()
        self.i_h = torch.linspace(0, o_w - 1, n_w).round().long()

        if self.cuda:
            self.i_v, self.i_h = self.i_v.cuda(), self.i_h.cuda()

        self.orig_size = orig_size
        self.new_size = new_size

    def __call__(self, orig_img, new_size):
        orig_size = orig_img.size()
        if not(self.orig_size == orig_size and self.new_size == new_size):
            self._compute_idxs(orig_size, new_size)
        return orig_img.index_select(1, self.i_v).index_select(2, self.i_h)



def main():
    orig_size, new_size = torch.Size([1, 257, 237]), torch.Size([1, 53, 35])

    bl_resizer = BilinearResizer(cuda=True,
                                 orig_size=orig_size, new_size=new_size)
    for _ in range(100):
        test_img = torch.randn(orig_size).cuda()
        bl_img = bl_resizer(test_img, new_size)
        assert bl_img.size() == new_size

    nn_resizer = NNResizer(cuda=True,
                           orig_size=orig_size, new_size=new_size)
    for _ in range(100):
        test_img = torch.randn(orig_size).cuda()
        nn_img = nn_resizer(test_img, new_size)
        assert nn_img.size() == new_size


if __name__ == "__main__":
    main()
