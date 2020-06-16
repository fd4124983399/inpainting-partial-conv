import torch

class SRMaskGenerator:
    def __init__(self, shape, device, sr_rate, dtype=torch.float):
        self.shape = shape
        self.img_shape = shape[1]
        self.device = device
        self.sr_rate = sr_rate
        self.dtype = dtype

    def get_sr_mask(self):
        sr_mask = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        index_h = 0
        index_v = 0
        while (index_v < self.img_shape):
            index_h = 0
            while (index_h < self.img_shape):
                sr_mask[0][index_v][index_h] = 1
                index_h += self.sr_rate
            index_v += self.sr_rate
        sr_mask = sr_mask.repeat(3, 1, 1)

        return sr_mask