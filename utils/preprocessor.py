import numpy as np
from PIL import Image


class Preprocessor(object):
    def __init__(self, env_type, target_size=(84, 84)):
        self.env_type = env_type
        self.target_size = target_size
        self.rgb_weigths = np.array([.2126, .7152, .0722], dtype=np.float32)

        # need to find a better way
        if self.env_type == "atari":
            self.transform = self._atari_preprocess
        elif self.env_type == "catch":
            self.transform = self._rgb2y

    def _rgb2y(self, o):
        return np.dot(o, self.rgb_weigths) / 255

    def _atari_preprocess(self, o):
        img = Image.fromarray(self._rgb2y(o))
        return np.array(img.resize(self.target_size, resample=Image.NEAREST))
