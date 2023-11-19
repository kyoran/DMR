import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, num_out_channels=9):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.init_height = 6
        self.init_width = 6
        self.num_out_channels = num_out_channels
        num_out_channels = num_out_channels
        # 9 rgb
        # 3 depth
        kernel = 3

        self.fc = nn.Linear(
            feature_dim, num_filters * self.init_height * self.init_width
        )

        self.deconvs = nn.ModuleList()

        # pads = [1, 0, 1]
        # pads = [0, 1, 0]
        # for i in range(self.num_layers - 1):
        #     output_padding = pads[i]
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, output_padding=0)
        )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, output_padding=0)
        )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, output_padding=0)
        )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_out_channels, 3, stride=2, output_padding=1)
        )

        self.outputs = dict()

    def forward(self, h):
        self.outputs['decoder_input'] = h

        h = torch.relu(self.fc(h))
        # self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.init_height, self.init_width)
        # self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            # self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['rec_obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        # for k, v in self.outputs.items():
        #     # L.log_histogram('train_decoder/%s_hist' % k, v, step)
        #     if len(v.shape) > 2:
        #         L.log_image(f'train_decoder_{self.num_out_channels}/%s' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param(
        #         'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
        #     )
        # L.log_param('train_decoder/fc', self.fc, step)


class pixelHybridEasy(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.init_height = 4
        self.init_width = 25

        rgb_num_out_channels = obs_shape[0][0]
        dvs_num_out_channels = obs_shape[1][0]

        num_out_channels = 9  # rgb
        kernel = 3

        self.rgb_fc = nn.Linear(
            feature_dim, num_filters * self.init_height * self.init_width
        )
        self.dvs_fc = nn.Linear(
            feature_dim, num_filters * self.init_height * self.init_width
        )

        pads = [0, 1, 0]

        self.rgb_deconvs = nn.ModuleList()
        self.dvs_deconvs = nn.ModuleList()
        for i in range(self.num_layers - 1):
            output_padding = pads[i]
            self.rgb_deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, kernel, stride=2, output_padding=output_padding)
            )
            self.dvs_deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, kernel, stride=2, output_padding=output_padding)
            )
        self.rgb_deconvs.append(
            nn.ConvTranspose2d(
                num_filters, rgb_num_out_channels, kernel, stride=2, output_padding=1
            )
        )
        self.dvs_deconvs.append(
            nn.ConvTranspose2d(
                num_filters, dvs_num_out_channels, kernel, stride=2, output_padding=1
            )
        )

        self.outputs = dict()

    def forward(self, h):
        rgb_h = torch.relu(self.rgb_fc(h))
        dvs_h = torch.relu(self.dvs_fc(h))
        self.outputs['rgb_fc'] = rgb_h
        self.outputs['dvs_fc'] = dvs_h

        rgb_deconv = rgb_h.view(-1, self.num_filters, self.init_height, self.init_width)
        dvs_deconv = dvs_h.view(-1, self.num_filters, self.init_height, self.init_width)
        self.outputs['rgb_deconv1'] = rgb_deconv
        self.outputs['dvs_deconv1'] = dvs_deconv

        for i in range(0, self.num_layers - 1):
            rgb_deconv = torch.relu(self.rgb_deconvs[i](rgb_deconv))
            dvs_deconv = torch.relu(self.dvs_deconvs[i](dvs_deconv))
            self.outputs['rgb_deconv%s' % (i + 1)] = rgb_deconv
            self.outputs['dvs_deconv%s' % (i + 1)] = dvs_deconv

        rgb_obs = self.rgb_deconvs[-1](rgb_deconv)
        dvs_obs = self.dvs_deconvs[-1](dvs_deconv)
        self.outputs['rgb_obs'] = rgb_obs
        self.outputs['dvs_obs'] = dvs_obs

        return [rgb_obs, dvs_obs]

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/rgb_deconv%s' % (i + 1), self.rgb_deconvs[i], step
            )
            L.log_param(
                'train_decoder/dvs_deconv%s' % (i + 1), self.dvs_deconvs[i], step
            )
        L.log_param('train_decoder/rgb_fc', self.rgb_fc, step)
        L.log_param('train_decoder/dvs_fc', self.dvs_fc, step)


class PixelMultiScaleDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.init_height = 6
        self.init_width = 6
        num_out_channels = 9  # rgb
        # num_out_channels = 3  # depth
        kernel = 3

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, padding=1))

        self.fc1 = nn.Linear(256 * 36, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)


        self.fc2 = nn.Linear(
            feature_dim, num_filters * self.init_height * self.init_width
        )

        self.deconvs = nn.ModuleList()
        # pads = [1, 0, 1]
        # pads = [0, 1, 0]
        # for i in range(self.num_layers - 1):
        #     output_padding = pads[i]
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, output_padding=0)
        )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, output_padding=0)
        )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, output_padding=0)
        )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_out_channels, 3, stride=2, output_padding=1)
        )

        self.outputs = dict()

    def forward(self, h):
        h1, h2, h3, h4 = h
        # h1.shape: torch.Size([32, 64, 62, 62])
        # h2.shape: torch.Size([32, 128, 30, 30])
        # h3.shape: torch.Size([32, 256, 14, 14])
        # h4.shape: torch.Size([32, 256, 6, 6])

        #
        # self.outputs['fc'] = h
        ###############################################
        curr_m = self.convs[0](h1)
        curr_m = curr_m + h2
        curr_m = self.convs[1](curr_m)
        curr_m = curr_m + h3
        curr_m = self.convs[2](curr_m)
        curr_m = curr_m + h4
        curr_m = self.convs[3](curr_m)
        ###############################################
        curr_m = curr_m.view(curr_m.size(0), -1)
        curr_m = self.ln(self.fc1(curr_m))
        h = torch.relu(self.fc2(curr_m))
        # h.shape: torch.Size([32, 1152])
        # print("h.shape:", h.shape)
        ###############################################
        deconv = h.view(-1, self.num_filters, self.init_height, self.init_width)
        # deconv.shape: torch.Size([32, 32, 6, 6])
        # self.outputs['deconv1'] = deconv
        # print("deconv.shape:", deconv.shape)

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            # self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['rec_obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param(
        #         'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
        #     )
        # L.log_param('train_decoder/fc', self.fc, step)

class HybridActionMaskV5Decoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.init_height = 6
        self.init_width = 6
        # num_out_channels = 9  # rgb
        num_out_channels = 3  # depth
        kernel = 3
        ###################
        self.curr_convs = nn.ModuleList()
        self.curr_convs.append(nn.Conv2d(3*1, 64, 5, stride=2))
        self.curr_convs.append(nn.Conv2d(64, 64, 3, stride=2))
        self.curr_convs.append(nn.Conv2d(64, 64, 3, stride=2))
        self.curr_convs.append(nn.Conv2d(64, 64, 3, stride=2))
        self.curr_fc = nn.Linear(64 * 36, feature_dim)
        ###################
        self.fc = nn.Linear(
            feature_dim * 2, num_filters * self.init_height * self.init_width
        )

        self.deconvs = nn.ModuleList()

        # pads = [1, 0, 1]
        # pads = [0, 1, 0]
        # for i in range(self.num_layers - 1):
        #     output_padding = pads[i]
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, output_padding=0)
        )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, output_padding=0)
        )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, output_padding=0)
        )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, num_out_channels, 3, stride=2, output_padding=1)
        )

        self.outputs = dict()

    def forward(self, rgb_h, dvs_h):

        # mask = self.curr_convs[0](mask)
        # mask = self.curr_convs[1](mask)
        # mask = self.curr_convs[2](mask)
        # mask = self.curr_convs[3](mask)
        # mask = mask.view(-1, 64 * 36)
        # mask = self.curr_fc(mask)
        ################################
        # h = torch.cat([mask, h], dim=1)
        fuse_h = torch.cat([rgb_h, dvs_h], dim=1)
        ################################
        h = torch.relu(self.fc(fuse_h))
        # self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.init_height, self.init_width)
        # self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            # self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['recon'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param(
        #         'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
        #     )
        # L.log_param('train_decoder/fc', self.fc, step)


_AVAILABLE_DECODERS = {
    'pixel': PixelDecoder,
    'pixelHybridEasy': pixelHybridEasy,
    'PixelMultiScaleDecoder': PixelMultiScaleDecoder,
    'HybridActionMaskV5Decoder': HybridActionMaskV5Decoder,
}


def make_decoder(
    decoder_type, obs_shape, feature_dim, num_layers, num_filters, num_out_channels
):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](
        obs_shape, feature_dim, num_layers, num_filters, num_out_channels
    )
