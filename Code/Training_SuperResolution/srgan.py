
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import L1Loss
import torchvision.models as models

class SR_ResidualBlk(nn.Module):
	def __init__(self, channels):
		super(SR_ResidualBlk, self).__init__()
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,padding=1)
		self.bn1 = nn.BatchNorm2d(channels)
		self.prelu = nn.PReLU()
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,padding=1)
		self.bn2 = nn.BatchNorm2d(channels)    

	def forward(self, x):
		residual = self.conv1(x)
		residual = self.bn1(residual)
		residual = self.prelu(residual)
		residual = self.conv2(residual)
		residual = self.bn2(residual)
		return x + residual

class SR_UpsampleBlk(nn.Module):
	def __init__(self, in_channels, up_scale):
		super(SR_UpsampleBlk, self).__init__()
		# [B, Res_C, H, W], where Res_C = residual_channels
		self.conv = nn.Conv2d(in_channels, in_channels * up_scale** 2, kernel_size=3, padding=1)
		# [B, Res_C * up_scale**2, H, W]
		self.pixel_shuffle = nn.PixelShuffle(up_scale)
		# [B, Res_C, H * up_scale, W * up_scale]
		self.prelu = nn.PReLU()
		
	def forward(self, x):
		x = self.conv(x)
		x = self.pixel_shuffle(x)
		x = self.prelu(x)
		return x

class SR_Generator(nn.Module):
	def __init__(self, scale_factor, residual_channels=25):
		upsample_block_num = int(math.log(scale_factor, 2))
		super(SR_Generator, self).__init__()
		self.block1 = nn.Sequential(nn.Conv2d(3, residual_channels, kernel_size=9, padding=4),nn.PReLU())	# only number of channel changes => ([B, residual_channels, H, W]). where H and W are crop_size // upscale_factor
		
		# gradually adding sprinkles of details on the top of the original feature map
		self.block2 = SR_ResidualBlk(residual_channels)
		self.block3 = SR_ResidualBlk(residual_channels)
		self.block4 = SR_ResidualBlk(residual_channels)
		self.block5 = SR_ResidualBlk(residual_channels)
		self.block6 = SR_ResidualBlk(residual_channels)
		# => the original feature map becomes 'dusted':
		self.block7 = nn.Sequential(nn.Conv2d(residual_channels, residual_channels, kernel_size=3, padding=1),nn.BatchNorm2d(25))
		
		block8 = [SR_UpsampleBlk(residual_channels, 2) for _ in range (upsample_block_num)]	# returning to the original dimensions (crop_scale x crop_scale)
		block8.append(nn.Conv2d(residual_channels, 3, kernel_size=9, padding=4))			# returning to the original depth (3 (RGB))
		self.block8 = nn.Sequential(*block8)	# the sequence: [UpsampleBlk, UpsampleBlk, UpsampleBlk, ..., UpsampleBlk, Conv2d], where pow(2, number_of_UpsampleBlk_s) == scale_factor
		
	def forward(self, x):
		# [B, 3, crop_size, crop_size]
		block1 = self.block1(x)
		# [B, 25, crop_size // upscale_factor, crop_size // upscale_factor] (same until after self.block7())
		block2 = self.block2(block1)
		block3 = self.block3(block2)
		block4 = self.block4(block3)
		block5 = self.block5(block4)
		block6 = self.block6(block5)
		block7 = self.block7(block6)
		# [B, 25, crop_size // upscale_factor, crop_size // upscale_factor]
		block8 = self.block8(block1 + block7)
		# [B, 3, crop_size, crop_size]
		return (torch.tanh(block8) + 1) / 2	# interval calculus (where tanh outputs between [-1, 1]): ([-1, 1] + 1) / 2 = [0, 1]
		# => all values of the returned generated image are 'lerped' between [0, 1]

class SR_Discriminator(nn.Module):
	def __init__(self):
		super(SR_Discriminator, self).__init__()
		self.net = nn.Sequential(
	# [B, C, crop_size, crop_size]
	nn.Conv2d(3, 25, kernel_size=3, padding=1),	nn.LeakyReLU(0.2),	nn.Conv2d(25, 25, kernel_size=3, stride=2, padding=1),	# [..., crop_size / 2, crop_size / 2]
	nn.BatchNorm2d(25),	nn.LeakyReLU(0.2),	nn.Conv2d(25, 50, kernel_size=3, padding=1),									# [..., crop_size / 2, crop_size / 2]
	nn.BatchNorm2d(50),	nn.LeakyReLU(0.2),	nn.Conv2d(50, 50, kernel_size=3, stride=2, padding=1),							# [..., crop_size / 4, crop_size / 4]
	nn.BatchNorm2d(50),	nn.LeakyReLU(0.2),	nn.Conv2d(50, 100, kernel_size=3, padding=1),									# [..., crop_size / 4, crop_size / 4]
	nn.BatchNorm2d(100),	nn.LeakyReLU(0.2),	nn.Conv2d(100, 100, kernel_size=3, stride=2, padding=1),					# [..., crop_size / 8, crop_size / 8]
	nn.BatchNorm2d(100),	nn.LeakyReLU(0.2),	nn.Conv2d(100, 200, kernel_size=3, padding=1),								# [..., crop_size / 8, crop_size / 8]
	nn.BatchNorm2d(200),	nn.LeakyReLU(0.2),	nn.Conv2d(200, 200, kernel_size=3, stride=2, padding=1),					# [..., crop_size / 16, crop_size / 16]
	nn.BatchNorm2d(200),	nn.LeakyReLU(0.2),	nn.AdaptiveAvgPool2d(1), # the arg (1) means the H and W dimension (H=W)	# [B, 200, 1, 1]
	nn.Conv2d(200, 400, kernel_size=1),	nn.LeakyReLU(0.2),	nn.Conv2d(400, 1, kernel_size=1))								# [B, 1, 1, 1]

	def forward(self, x):
	    batch_size = x.size(0)
	    return torch.sigmoid(self.net(x).view(batch_size))	# [B, 1, 1, 1] => [B]

class SR_GeneratorLoss(nn.Module):
	def __init__(self):
		super(SR_GeneratorLoss, self).__init__()
		vgg = models.vgg16(pretrained=True)
		# for visualising the vgg model structure, check the following screenshot in this folder: the_sequence_of_vgg16_layers.png
		# in total: (13 x Conv2D(), 13 x ReLU(), 5 x MaxPool2D())
		# as it looks like, it generates a very deep feature map ( depth: 512, section area: H = W = input_size // (2^5) )
		loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
		for param in loss_network.parameters():
			param.requires_grad = False
		self.loss_network = loss_network
		self.mse_loss = nn.MSELoss() 			
		self.l2_loss = L1Loss()
		
	def forward(self, out_labels, out_images, target_images):
		# Calculating loss values for each batch. The loss of one batch is equal to the mean of the losses for each image in the batch. 
		# Thus, a loss is just a float number, because it is the mean of loss values from a batch of shape [B].
		# There are 4 loss functions for each of which we calculate the loss values:

		#adversarial Loss
		adversarial_loss = torch.mean(1 - out_labels)
		#vgg Loss
		vgg_loss = self.mse_loss(self.loss_network(out_images),
		self.loss_network(target_images))
		#pixel-wise Loss
		pixel_loss = self.mse_loss(out_images, target_images)
		#regularization Loss
		reg_loss = self.l2_loss(out_images, target_images)
		return pixel_loss + 0.001 * adversarial_loss + 0.006 * vgg_loss + 2e-8 * reg_loss

# this class (loss function) seems to be unused in this project
class SR_L2Loss(nn.Module):
	def __init__(self, l2_loss_weight=1):
		super(SR_L2Loss, self).__init__()
		self.l2_loss_weight = l2_loss_weight
		
	def forward(self, x):
		batch_size = x.size()[0]
		h_x = x.size()[2]
		w_x = x.size()[3]
		count_h = self.tensor_size(x[:, :, 1:, :])
		count_w = self.tensor_size(x[:, :, :, 1:])
		h_l2 = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
		w_l2 = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
		return self.l2_loss_weight * 2 * (h_l2 / count_h + w_l2 / count_w) / batch_size
		
	@staticmethod
	def tensor_size(t):
		return t.size()[1] * t.size()[2] * t.size()[3]

