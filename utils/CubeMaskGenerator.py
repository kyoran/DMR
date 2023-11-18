import torch.nn as nn
import torch
import numpy as np

class CubeMaskGenerator:
    def __init__(self, input_size, image_size, clip_size, block_size, mask_ratio):
        assert mask_ratio <= 1.0

        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.image_size = image_size
        self.upsampler = nn.Upsample((image_size, image_size))

        self.block_size = block_size
        self.num_blocks = clip_size // block_size
        print("\t@@@@@@@@@@: clip_size", clip_size)         # 6+1 = 7
        print("\t@@@@@@@@@@: block_size", block_size)       # 4
        print("\t@@@@@@@@@@: num_blocks", self.num_blocks)  # 1

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        # print("!!!!!!!!:", mask.shape)
        cube_mask = None
        for i in range(self.num_blocks):
            np.random.shuffle(mask)
            cur_mask = torch.from_numpy(mask).reshape(self.height, self.width)
            cur_mask = self.upsampler(cur_mask[None, None].float())  # (1, 1, h, w)
            cur_mask = cur_mask.expand(self.block_size, *cur_mask.size()[1:])
            cube_mask = torch.cat([cube_mask, cur_mask]) if i > 0 else cur_mask
            # print("\t\tcube_mask:", cube_mask.shape)
        return cube_mask
