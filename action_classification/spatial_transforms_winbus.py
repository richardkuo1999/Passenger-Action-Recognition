import random
import math
import numbers
import collections
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2

try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms
        #self.indexx = 0
    def __call__(self, img):
        #xx = 0
        for t in self.transforms:
            img = t(img)
            #if xx==0:
            #    img.save('E:/3dres4class/ex/%05d.jpg'%self.indexx)
            #    self.indexx+=1
            #xx+=1
        return img

    def randomize_parameters(self):
        pass


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class ColorAugment(object):
    def __init__(self):
        #self.a = np.random.uniform(-1, 1, 3) * [0.0138, 0.678, 0.36] + 1
        self.a = np.random.uniform(-1, 1, 3) * [0.2, 0.8, 0.38] + 1
        self.CAindex = 0

    def __call__(self, img):
        # Augment colorspace
        # SV augmentation by 50%
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

        img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * self.a).clip(None, 255).astype(np.uint8)
        np.clip(img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0])  # inplace hue clip (0 - 179 deg)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


        if self.CAindex == 15-1:
            self.a = np.random.uniform(-1, 1, 3) * [0.2, 0.8, 0.38] + 1
            self.CAindex = 0
        else:
            self.CAindex += 1

        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        return img

    def randomize_parameters(self):
        self.p = random.random()


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""
    def __init__(self):
        self.pp = None
        self.RHFindex = 0

    def __call__(self, img):
        if self.RHFindex == 15:
            self.pp = random.random() < 0.5
            self.RHFindex = 0
        if self.pp:
            self.RHFindex += 1
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            self.RHFindex += 1
            return img

    def randomize_parameters(self):
        pass

class RandomVerticalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def randomize_parameters(self):
        self.p = random.random()
