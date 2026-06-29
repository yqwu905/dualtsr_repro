import torch
import torch.nn as nn
from .rdp_unet import RdpUnet
from types import SimpleNamespace
from safetensors.torch import load_file


class EncoderX_f8c32(nn.Module): # lite
    def __init__(self, rdp_num=10, use_checkpoint=False):
        super().__init__()
        self._tied_weights_keys = []
        self.npu_conv_in = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.rdpmodules = nn.Sequential(*[RdpUnet(use_checkpoint=use_checkpoint) for _ in range(rdp_num)])
        self.down = nn.Sequential(nn.Conv2d(16, 16*2, kernel_size=3, stride=2, padding=1),
                                  nn.Conv2d(16*2, 16*4, kernel_size=3, stride=2, padding=1),
                                  nn.Conv2d(16*4, 16*8, kernel_size=3, stride=2, padding=1),
                                  )
        self.npu_conv_out = nn.Conv2d(16*8, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.npu_conv_in(x)
        x = self.rdpmodules(x)
        x = self.down(x)
        x = self.npu_conv_out(x)
        return x

class DecoderX_f8c32(nn.Module): #  lite
    def __init__(self, rdp_num=10, use_checkpoint=False):
        super().__init__()
        self._tied_weights_keys = []
        channel = 128
        self.npu_conv_in = nn.Conv2d(32, channel, kernel_size=3, stride=1, padding=1)
        self.up = nn.Sequential(nn.ConvTranspose2d(channel, channel//2, kernel_size=2, stride=2, bias=True),
                                nn.ConvTranspose2d(channel//2, channel//4, kernel_size=2, stride=2, bias=True),
                                nn.ConvTranspose2d(channel//4, 16, kernel_size=2, stride=2, bias=True),
                                )
        self.rdpmodules = nn.Sequential(*[RdpUnet(use_checkpoint=use_checkpoint) for _ in range(rdp_num)])
        self.npu_conv_out = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.npu_conv_in(x)
        x = self.up(x)
        x = self.rdpmodules(x)
        x = self.npu_conv_out(x)
        return x
    def get_last_layer(self):  # for gan
        return self.npu_conv_out.weight


class VAE16X(nn.Module):
    def __init__(self, ckpt_path=None, use_checkpoint=False):
        super(VAE16X, self).__init__()
        self.encoder = EncoderX_f8c32(use_checkpoint=use_checkpoint)
        self.decoder = DecoderX_f8c32(use_checkpoint=use_checkpoint)
        if ckpt_path is not None:
            print('*************** loading ***************')
            if ckpt_path.endswith(".safetensors"):
                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(ckpt_path, weights_only=True, map_location='cpu')
            miss, unexpect = self.load_state_dict(state_dict, strict=False)
            if len(miss) != 0 or len(unexpect) != 0:
                print(miss)
                print(unexpect)
            print('*************** loading done ***************')

        self.config = SimpleNamespace()
        self.config.shift_factor = 0.07050679
        self.config.scaling_factor = 0.2517327

    def forward(self, x):
        latent = self.encoder(x)
        rec = self.decoder(latent)
        return latent, rec

    def encode(self, x, return_dict=True):
        res = self.encoder(x)
        if not return_dict:
            return (res,)
        return res
    def decode(self, x, return_dict=True):
        res = self.decoder(x)
        if not return_dict:
            return (res,)
        return res



if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms.functional as tf
    from torchvision.utils import save_image
    model = VAE16X('ckpt/f8c32/20260120/nch_f8c32_checkpoint-step-0012600_ema.pth').to(device='cuda')
    model.eval()
    image = Image.open('test_images/2048.png').convert('RGB')
    image_tensor = tf.to_tensor(image)[None] * 2 - 1 # [-1 ,1]
    with torch.no_grad():
        latent, rec = model(image_tensor.to(device='cuda'))
    save_image(rec / 2 + 0.5, 'test_images/result_ch32_8x.png')
