from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils

from utils.tools import extract_image_patches, flow_to_image, \
    reduce_mean, reduce_sum, default_loader, same_padding

def gpu_usage(msg):
    a = torch.cuda.memory_allocated(0)
    print('for ', msg, ':', a)


class Generator(nn.Module):
    def __init__(self, config, use_cuda, device_ids):
        super(Generator, self).__init__()
        self.input_dim = config['netG']['input_dim']
        self.cnum = config['netG']['ngf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum, config, self.use_cuda, self.device_ids)
        self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2, offset_flow


class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum, config, use_cuda=True, device_ids=None):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        # self.coarse_ones = torch.ones(config['batch_size'], 1, config['image_shape'][0], config['image_shape'][1])
        # self.coarse_mask = torch.ones(config['batch_size'], 1, config['image_shape'][0], config['image_shape'][1])
        # if self.use_cuda:
        #     self.coarse_ones = self.coarse_ones.cuda()
        #     self.coarse_mask = self.coarse_mask.cuda()

        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        self.conv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        self.conv11 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.conv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')
        # all these filters weigh about 6MB

    def forward(self, x, mask):
        # For indicating the boundaries of images
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()

        # ones and mask weigh in total 5MB
        
        x = self.conv1(torch.cat([x, ones, mask], dim=1))
        # 5 x 256 x 256
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest', recompute_scale_factor=True)
        # cnum*4 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest', recompute_scale_factor=True)
        # cnum*2 x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256
        x_stage1 = torch.clamp(x, -1., 1.)

        return x_stage1

    # def revert_coarse_tensors(self):
    #     for _ in self.coarse_ones:
    #         for _ in self.coarse_ones:
    #             for ones_height in self.coarse_ones:
    #                 for i in range(0, ones_height.size(0)):
    #                     ones_height[i] = 1


class FineGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(FineGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        self.conv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        # cnum*2 x 128 x 128
        self.conv4_downsample = gen_conv(cnum*2, cnum*2, 3, 2, 1)
        self.conv5 = gen_conv(cnum*2, cnum*4, 3, 1, 1)
        # cnum*4 x 64 x 64
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)


        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1, activation='relu')
        self.contextul_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True, use_cuda=self.use_cuda, device_ids=self.device_ids)
        self.pmconv9 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv11 = gen_conv(cnum*8, cnum*4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # conv branch
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest', recompute_scale_factor=True)
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest', recompute_scale_factor=True)
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1., 1.)

        return x_stage2, offset_flow


class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False, use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    # f, b [batch_size, C = 128, H = 64, W = 64]
    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        #if not mask == None:
        #    print('mask shape = ', mask.shape)

        # get shapes
        raw_int_fs = list(f.size())   # b*c*h*w (batch_size * chaannels * height * width)
        raw_int_bs = list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L], in general L = H * W (1024 = 32 * 32)
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0) # raw_w_groups = tuple of N groups of L patches (each patch group being [L,C,k,k], meaning L patches of size [C,k,k]), where N is the batch_size. So the length of the tuple is N

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest', recompute_scale_factor=True)
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest', recompute_scale_factor=True)
        int_fs = list(f.size())     # b*c*h*w (H = 32, W = 32)
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0) # (similar to raw_w_groups) w_groups = tuple of N groups of L patches (each patch group being [L,C,k,k], meaning L patches of size [C,k,k]), where N is the batch_size. So the length of the tuple is N

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])    # one single channel, the other dimensions are the same [B, 1, H, W]
            if self.use_cuda:
                mask = mask.cuda()
        else:
            # [B, 1, 256, 256] => [B, 1, 32, 32]
            mask = F.interpolate(mask, scale_factor=1./(4*self.rate), mode='nearest', recompute_scale_factor=True)
        int_ms = list(mask.size())  # [B, 1, 32, 32]

        # mask shape is [B, 1, H, W], for example [2, 1, 32, 32]

        # m shape: [N, C*k*k, L] # ???
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        # m.shape = [2, 1024, 1, 3, 3]
        m = m[0]
        # m.shape = [1024, 1, 3, 3]

        # m shape: [L, C, k, k]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(torch.float32) # the average value of all the values in the last 3 dimensions
        # for the shape [L, C, k, k] the last 3 dimensions are [C, k, k].
        # so, for example, if C = 3, k = 3, k = 3 (total number of values = 3*3*3=27), and let's say all the values are 7, then the reduce_mean() places "7" instead of the [C,k,k] 3D tensor. 
        # So instead of a [C,k,k] tensor, it would still be a 3D tensor, but a [1,1,1], containing only the value "7" (obviously, at index [0][0][0])

        # mm shape: [L, 1, 1, 1]
        mm = mm.permute(1, 0, 2, 3) 
        # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k  # torch.eye(k) is the 2D identity matrix of k lines and k cols
        # print('fuse_weight: ', fuse_weight)
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        # iteram pe patch-uri (un patch per iteratie):
        # (downscaled foreground (32x32) | patches of downscaled background (1024=32x32) | patches of background (1024=64x64 2-strided))
        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            # print('wi_normed shape = ', wi_normed.shape)
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # f convolved using patches from b as filters => yi

            # FUSION: ATTENTION PROPAGATION in the horizontal and vertical neighbourhoods (within a [self.fuse_k / 2 - 1] range from the original point)
            # conv implementation for fuse scores to encourage large patches
            # basically, fuse applies convolution with I3 (2d identity matrix 3x3) to the whole HxW "image", for all the "images" in yi (L "images" in total)
            # input to conv has 1 channel, output of conv also has 1 channel
            # it does this same thing twice, in the following "if" branch:
            # the issue is: the identity matrix has values of 1 only in the diagonal, but we want to check for neighbourhood horizontally and then vertically, NOT diagonally, so it does not really make sense to use torch.eye() as convolution kernel
            if self.fuse:
                # Not sure I understand the permutation of dimensions below. L dimension is split in widths and heights, but is this really the right order of each patch's spatial locality?
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                # idk what is this permutation for, but I guess we want to switch width and height dimensions in order to switch from horizontal neighbourhood check to vertical neighbourhood check (but not sure)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # back to [1, L, H, W]

            # element-wise multiplication:
            # tst_2 = torch.Tensor([[6, 3, 4], [8, 13, 10]])
            # tst_3 = torch.Tensor([[2, 3, 5], [7, 3, 4]])
            # tst_5 = torch.Tensor([[2]])
            # tst_7 = torch.Tensor([[1, 2]])
            # tst_4 = tst_2 * tst_3
            # => tensor(
            #   [[12.,  9., 20.],
            #   [56., 39., 40.]]
            # )
            # tst_6 = tst_2 * tst_5
            # => [[12, 6, 8], [16, 26, 20]]
            # tst_8 = tst_2 * tst_7
            # => error (sizes not corresponing and/or not 1) - so the multiplication does not make sense in this situation

            # softmax to match
            yi = yi * mm    # ELEMENT-WISE multiplication!  # [1, L, H, W] * [1, L, 1, 1] = [1, L, H, W]. If yi and mm don't have same dimensions, element-wise is NOT possible, unless one of them has all sizes equal to 1 (the sizes which get multiplied). Example a few lines above ^.
            # all elements of yi remain unchanged, except the ones where mm is 0 (0 = masked, 1 = not masked)
            # basically, all patches of yi under the mask become null patches (fulfilled with zeros), and any other patch remains unchanged
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]    # ELEMENT-WISE multiplication! again!

            # Why is there need for 2 multiplications? (yi * mm twice) 
            # Because we filter the yi by the mask. We are considering only the masked image. 
            # Then, after the softmax function (which averages each value in the image),
            # there are ALSO the pixels within the mask which get averaged (and thus, != 0). 
            # This is why we need to apply the mask again, so that elements under the mask stay at 0.

            

            # argmax in a tensor of shape [1, L, H, W], with dim=1. dim=1 takes the second dimension of the tensor (the dimension with the length L).
            # The output is of shape [1, 1, H, W].
            # Let's consider a tensor of [1, N, H, W] the following range of addresses in the 4D tensor: [1][X][12][19], where x is variable, x < N
            # Then, the result of argmax is a [1, 1, H, W] tensor which, on the [1][1][12][19] position, it will have the index X of maximum number of the above range,
            # meaning: X where the value at [1][X][12][19] is maximum, where X can be anything in interval [0...L-1]
            # REMEMBER: in a tensor of shape [S1, S2, ... SN], dim=0 refers to S1, dim=1 refers to S2, ... dim=N-1 refers to SN
            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*Hf*Wf (Hf = H_foreground, Wf = W_foreground)
            # => se ia fiecare h <- [0, ..., H-1] si w <- [0, ..., W-1], si pentru fiecare adresa [1][1][h][w] se scrie indexul L_mic <- [0, ..., L-1] al valoarii maxime de pe locul [1][L_mic][h][w].
            # => in offset se retine patch-ul (din background) in convolutie cu care pixelul curent (din froeground) de la (h, w) se potriveste cel mai bine

            # argmax test (uncommented the lines below if you want to see how it works in practice):
            # tst_0 = torch.Tensor([[[10, 2, 3], [11, 6, 7]], [[2, 3, 5], [1, 7, 6]]])
            # tst_1 = torch.argmax(tst_0, dim=0, keepdim=True)
            # tst_2 = torch.argmax(tst_0, dim=1, keepdim=True)
            # print('tst_1 = ', tst_1)
            # print('tst_2 = ', tst_2)

            #print('offset = ', offset)

            # sunt egale ([2, 128, 32, 32]), deci nu se va intra in acest if.
            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])     # ratio between: area of the downscaled foreground | area of the downscaled background (whole background, including mask area)
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            # offset = torch.cat([offset//int_fs[3], offset%int_fs[3]], dim=1)  # 1*2*H*W
            offset = torch.cat([torch.div(offset, int_fs[3], rounding_mode='trunc'), offset%int_fs[3]], dim=1)  # 1*2*H*W
            # print('offsett = ', offset)

            # inputs = torch.randn(1, 4, 5, 5)
            # weights = torch.randn(4, 8, 3, 3)
            # dds = F.conv_transpose2d(inputs, weights, padding=0)
            # print('DDS SHAPE =', dds.shape)

            # filters = torch.randn(8, 4, 3, 3)
            # inputs = torch.randn(1, 4, 5, 5)
            # dds1 = F.conv2d(inputs, filters, padding=0)
            # print('DDS1 SHAPE =', dds1.shape)

            # yi.shape = [1, 1024, 32, 32]
            # raw_wi shape = [1, 1024, 128, 4, 4]

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            # deconvolution with the 64x64 (prior-downscaling) background => dilated image with same dimensions as the prior-downscaling one
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        # torch.cat(...,dim=0) face din lista de tensori un nou tensor de tensori
        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs) # nu face nimic ca y avea deja shape-ul egal cu raw_int_fs

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])   #  => shape: [batch_size, 2, 32, 32]

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2]).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3]) # [[0, 0, 0], [1, 1, 1], [2, 2, 2], ...]
        w_add = torch.arange(int_fs[3]).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1) # [[0, 1, 2, ...], [0, 1, 2, ...], [0, 1, 2, ...]]
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        if self.use_cuda:
            ref_coordinate = ref_coordinate.cuda()

        offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2) # reverted permutation from above
        if self.use_cuda:
            flow = flow.cuda()
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate*4, mode='nearest', recompute_scale_factor=True)

        return y, flow


def test_contextual_attention(imageA, imageB, imageOut):
    # import cv2
    import os
    # run on cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    def float_to_uint8(img):
        img = img * 255
        return img.astype('uint8')

    rate = 2
    stride = 1
    grid = rate*stride

    b = default_loader(imageA)
    w, h = b.size
    b = b.resize((w//grid*grid//2, h//grid*grid//2), Image.ANTIALIAS)
    # b = b.resize((w//grid*grid, h//grid*grid), Image.ANTIALIAS)

    f = default_loader(imageB)
    w, h = f.size
    f = f.resize((w//grid*grid, h//grid*grid), Image.ANTIALIAS)

    f, b = transforms.ToTensor()(f), transforms.ToTensor()(b)
    f, b = f.unsqueeze(0), b.unsqueeze(0)
    if torch.cuda.is_available():
        f, b = f.cuda(), b.cuda()

    contextual_attention = ContextualAttention(ksize=3, stride=stride, rate=rate, fuse=True)

    if torch.cuda.is_available():
        contextual_attention = contextual_attention.cuda()

    yt, flow_t = contextual_attention(f, b)
    vutils.save_image(yt, 'vutils' + imageOut, normalize=True)
    vutils.save_image(flow_t, 'flow' + imageOut, normalize=True)
    # y = tensor_img_to_npimg(yt.cpu()[0])
    # flow = tensor_img_to_npimg(flow_t.cpu()[0])
    # cv2.imwrite('flow' + imageOut, flow_t)


class LocalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(LocalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*8*8, 1)

    def forward(self, x):
        # [B*2, 3, 128, 128]
        x = self.dis_conv_module(x)
        # [B*2, 256, 8, 8]
        x = x.view(x.size()[0], -1)
        # [B*2, cnum*4*8*8], cnum = 64
        x = self.linear(x)
        # [B*2, 1]

        # De ce B*2? Pentru ca input-ul (x) este un tensor ce contine batch-ul cu ground truth ([B, ...]), concatenat cu rezultatele de la inpaint ([B, ...]) => ([B*2, ...])

        return x


class GlobalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):
        # [B, 3, 256, 256]
        x = self.dis_conv_module(x)
        # [B, 256, 16, 16]
        x = x.view(x.size()[0], -1)
        # [B, cnum*4*16*16]
        x = self.linear(x)
        # [B, 1]

        return x


class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)