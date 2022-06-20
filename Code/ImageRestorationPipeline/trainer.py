import torch
import torch.nn as nn
from inpainting_model.networks import Generator

from inpainting_utils.logger import get_logger

logger = get_logger()


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.use_cuda = config['cuda']
        self.netG = Generator(config, self.use_cuda)
        self.device = torch.device("cuda:{}".format(config['gpu_ids'][0]))

        if self.use_cuda:
            self.netG.to(self.device)

    # for the current batch:
    def forward(self, x, masks):
        self.eval()

        x1, x2, offset_flow = self.netG(x, masks)
        x2_inpaint = x2 * masks + x * (1. - masks)

        return x2_inpaint, offset_flow