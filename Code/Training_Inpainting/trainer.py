import os
import torch
import torch.nn as nn
from torch import autograd
from model.networks import Generator, LocalDis, GlobalDis
import torchvision.utils as vutils


from utils.tools import get_model_list, local_patch, spatial_discounting_mask
from utils.logger import get_logger

logger = get_logger()

def gpu_usage(msg):
    a = torch.cuda.memory_allocated(0)
    print('for ', msg, ':', a)


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.device = torch.device("cuda:{}".format(config['gpu_ids'][0]))

        self.netG = Generator(self.config, self.use_cuda, self.device_ids)
        self.localD = LocalDis(self.config['netD'], self.use_cuda, self.device_ids)
        self.globalD = GlobalDis(self.config['netD'], self.use_cuda, self.device_ids)

        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        d_params = list(self.localD.parameters()) + list(self.globalD.parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        if self.use_cuda:
            self.netG.to(self.device)
            self.localD.to(self.device)
            self.globalD.to(self.device)

    # for the current batch:
    def forward(self, x, bboxes, masks, ground_truth, compute_loss_g=False):
        #
        self.train()
        l1_loss = nn.L1Loss()
        losses = {}

        # masks -- values between [0, 1]
        # x -- values between [-1, 1], in fact min -1, max 0.8667


        x1, x2, offset_flow = self.netG(x, masks)
        local_patch_gt = local_patch(ground_truth, bboxes)  # image under the mask (the size is the size of the mask)
        x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)
        local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)
        local_patch_x2_inpaint = local_patch(x2_inpaint, bboxes)

        # D part
        # wgan d loss (the discriminator wants to minimize the difference between evaluating the gt and evaluating the masked gt). The difference is computed by losses['wgan_d']. 
        # Then, differences from all samples in the current batch are averaged.
        # There's a critic output for each evaluation (it's not 0 or 1 as in the general classifier discriminator, because this is an art critic discriminator). 
        # So the evaluation (of the gt or of the generated) can be any number, between -inf and inf. 
        # But the difference between the two evaluations is enhanced by the discriminator and reduced by the generator.
        # Note: x1 is NOT relevant for the discriminator, only x2 is.
        # Note: the inpaint results below are .detach()-ed because D does not want to track how G generated the image. D's improvement does not depend on how G performed.
        #   And tensor.detach() returns a new copy of the tensor, thus not an in-place operation. So the inpainted results will have tracked gradients on the G side (G part).
        local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
            self.localD, local_patch_gt, local_patch_x2_inpaint.detach())   # ([B, 1], [B, 1])
        global_real_pred, global_fake_pred = self.dis_forward(
            self.globalD, ground_truth, x2_inpaint.detach())                # ([B, 1], [B, 1])
        # if by torch.mean() you only specify the tensor, not the dimension, it will calculate the mean of all values on all dimensions, returning a 1D and 1-size tensor (or simply, a number)
        # Since we need to maximize the (pred_real - pred_fake) loss, we want our optimizer to minimize the negative of that loss, which is (pred_fake - pred_real).
        losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
            torch.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']
        # gradients penalty loss
        local_penalty = self.calc_gradient_penalty(
            self.localD, local_patch_gt, local_patch_x2_inpaint.detach())
        global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth, x2_inpaint.detach())
        losses['wgan_gp'] = local_penalty + global_penalty

        # G part
        # Note: below, results from inpaintings have tracked gradients (which won't be detached copies like above), 
        # because here we want to update the G's parameters based on gradients obtained by the performed generation process.
        if compute_loss_g:
            sd_mask = spatial_discounting_mask(self.config)
            vutils.save_image(local_patch_gt * sd_mask, 'generated_13.png', nrow=3 * 4, normalize=True) # ONLY for visualising a masked image! It is not used in the inpainting process, but only in GAN loss functions.
            # sd_mask helps only on calculating the l1_loss, only LOCALLY (on the mask's span). 
            # What is does is preventing pixels in the middle of the mask to seriously affect the l1 loss 
            # (the l1 loss between the inpaint and the ground truth)
            # The pixels that matter the most are the ones near the edges of the mask. The latter ones are weighted asimptotically lower, in order for the matter to decrease.
            # Wonder why? Because there's no real interest of what's in the middle of the mask, but the pixels near the edge are the ones that help a lot for the content-aware filling.

            # local:
            losses['l1'] = l1_loss(local_patch_x1_inpaint * sd_mask, local_patch_gt * sd_mask) * \
                self.config['coarse_l1_alpha'] + \
                l1_loss(local_patch_x2_inpaint * sd_mask, local_patch_gt * sd_mask)
            # global (excluding mask range):
            losses['ae'] = l1_loss(x1 * (1. - masks), ground_truth * (1. - masks)) * \
                self.config['coarse_l1_alpha'] + \
                l1_loss(x2 * (1. - masks), ground_truth * (1. - masks))

            # wgan g loss (only x2 matters here, x1 is irrelevant)
            # G wants to MAXIMIZE the _pred results, in order for the loss function to be minimum.
            # local:
            local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
                self.localD, local_patch_gt, local_patch_x2_inpaint) # same inference as the first one on the D side
            # global:
            global_real_pred, global_fake_pred = self.dis_forward( # same inference as the second one on the D side
                self.globalD, ground_truth, x2_inpaint)
            losses['wgan_g'] = - torch.mean(local_patch_fake_pred) - \
                torch.mean(global_fake_pred) * self.config['global_wgan_loss_alpha']

        return losses, x2_inpaint, offset_flow

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)
        # shapes of real_pred and of fake_pred: [B, 1]

        return real_pred, fake_pred

    # Calculate gradient penalty
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]
        # the gradient of disc_interpolates is taken into consideration (the gradient was tracked along netD() calculations)

        # torch.norm(tensor, dim=i) supresses the specified dimension of the given tensor. [d1][d2]...[di]...[dn] => pe pozitia [d1][d2]...no_di...[dn] se pune norma array-ului format din toate elementele de pe dimensiunea di, cu acelasi [d1][d2]...[dn]
        # c = torch.tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= torch.float)
        # torch.norm(c, dim=0) => tensor([sqrt(1^2 + (-1)^2), sqrt(2^2 + 1^2), sqrt(3^2 + 4^2)])
        # torch.norm(c, dim=1) => tensor([sqrt(1^2 + 2^2 + 3^2), sqrt((-1)^2 + 1^2 + 4^2)])

        # c = torch.tensor([\
        #     [[ 1, 2, 3], \
        #     [-1, 1, 4]], \
        #     \
        #     [[3, 2, 4], \
        #     [6, 1, 4]]] , dtype= torch.float)
        # print('dim0 = ',torch.norm(c, dim=0)**2) => tensor([[10.0000,  8.0000, 25.0000], [37.0000,  2.0000, 32.0000]])
        # print('dim1 = ',torch.norm(c, dim=1)**2) => tensor([[ 2.0000,  5.0000, 25.0000], [45.0000,  5.0000, 32.0000]])
        # print('dim2 = ',torch.norm(c, dim=2)**2) => tensor([[14.0000, 18.0000], [29.0000, 53.0000]])
        
        gradients = gradients.view(batch_size, -1) # [B, C*H*W], C = number of channels, (H, W) = height, width of the local_patch (128, 128)
        #batch_gradient_penalties = (torch.norm(gradients, dim=1) - 1) ** 2 # average of C*H*W (for each batch) => shape [B]
        batch_gradient_penalties = (gradients.norm(2, dim=1) - 1) ** 2
        gradient_penalty = batch_gradient_penalties.mean() # arithmetic mean over the values in [B] => a single value output for gradient_penalty

        return gradient_penalty

    def inference(self, x, masks):
        self.eval()
        x1, x2, offset_flow = self.netG(x, masks)
        # x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)

        return x2_inpaint, offset_flow

    def save_model(self, checkpoint_dir, iteration):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
        dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(self.netG.state_dict(), gen_name)
        torch.save({'localD': self.localD.state_dict(),
                    'globalD': self.globalD.state_dict()}, dis_name)
        torch.save({'gen': self.optimizer_g.state_dict(),
                    'dis': self.optimizer_d.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, iteration=0, test=False):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
        self.netG.load_state_dict(torch.load(last_model_name))
        iteration = int(last_model_name[-11:-3])

        if not test:
            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis", iteration=iteration)
            state_dict = torch.load(last_model_name)
            self.localD.load_state_dict(state_dict['localD'])
            self.globalD.load_state_dict(state_dict['globalD'])
            # Load optimizers
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.optimizer_d.load_state_dict(state_dict['dis'])
            self.optimizer_g.load_state_dict(state_dict['gen'])

        print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        logger.info("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        return iteration