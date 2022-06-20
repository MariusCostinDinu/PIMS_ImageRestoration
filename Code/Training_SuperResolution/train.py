import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
#from loss import GeneratorLoss
#from model import Generator, Discriminator

# from loss import GeneratorLoss
# from model import Generator, Discriminator
from srgan import SR_GeneratorLoss, SR_Discriminator, SR_Generator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=2, type=int, help='train epoch number')
parser.add_argument('--save_images_at_epoch', default=2, type=int, help='save images when nb of epochs is multiple of this')
parser.add_argument('--image_save_sparsity', default=20, type=int, help='save only some images, using this step value. If this is 1, save all images')
parser.add_argument('--save_params_at_epoch', default=100, type=int, help='save network paramteres when number of epochs is multiple of this')
parser.add_argument('--resume_params', default=True, type=bool, help='resume net config')
parser.add_argument('--resume_from_epoch', default=20, type=int, help='epoch id to resume from')
        


if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    
    train_set = TrainDatasetFromFolder('Images200x200', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('Images400x400', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    netG = SR_Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = SR_Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = SR_GeneratorLoss()
    
    device = torch.device('cpu')
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
        device = torch.device('cuda:0')
    
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    # 'epochs/netG_epoch_%d_%d.pth'

    if opt.resume_params:
        loaded_netG_weights = torch.load('saved_nets/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, opt.resume_from_epoch))
        netG_weights = {}
        for k, v in loaded_netG_weights.items():
            netG_weights[k] = v.to(device)
        netG.load_state_dict(netG_weights)

        loaded_netD_weights = torch.load('saved_nets/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, opt.resume_from_epoch))
        netD_weights = {}
        for k, v in loaded_netD_weights.items():
            netD_weights[k] = v.to(device)
        netD.load_state_dict(netD_weights)

    # tst_0 = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12])
    # tst_1 = torch.tensor([[1,2],[3,4]])
    # tst_0 = tst_0.reshape(2, 6)
    # print('reshape1 ', tst_0)
    # tst_0 = tst_0.reshape(2, 3, 2)
    # print('reshape2 ', tst_0)
    # #tst_0 = tst_0.reshape(2, 2, 3)
    # #print('reshape2 ', tst_0)
    # tst_0 = tst_0.permute(0, 2, 1)
    # print('permute tst0 ', tst_0)
    # tst_0 = tst_0.reshape(tst_0.shape[0], -1)
    # print('viewed tst0 ', tst_0)
    # print('permute ', tst_1.permute(1,0))

    # val_bar = tqdm(val_loader)
    # for val_lr, val_hr_restore, val_hr in val_bar:
    #     print('shapes :')
    #     print(val_lr.shape)
    #     print(val_hr_restore.shape)
    #     print(val_hr.shape)
    #     break
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        # iterate over all dataset, with B images at once, so the number of iterations is (1 + dataset.size // B)
        # FYI, dataset with dogs has size 527
        for data, target in train_bar:
            # data      [B, C, CROP_SIZE // upscale_factor, CROP_SIZE // upscale_factor]
            # target    [B, C, CROP_SIZE, CROP_SIZE]
            g_update_first = True
            batch_size = data.size(0)   # B (B is equal to 64, except last batch which might be smaller => of size equal to dataset.size() % B)
            running_results['batch_sizes'] += batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target) # [B, C, CROP_SIZE, CROP_SIZE], does Variable() even do anything?
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)          # [B, C, CROP_SIZE // upscale_factor, CROP_SIZE // upscale_factor]
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)          # [B, C, CROP_SIZE, CROP_SIZE]
    
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            ##

            # def forward(self, out_labels, out_images, target_images):
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            
            
            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        valing_results = None

        if epoch % opt.save_images_at_epoch == 0:
            netG.eval() # switching from training mode to test mode
            out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
                val_images = []
                index = 0
                for val_lr, val_hr_restore, val_hr in val_bar:
                    index += 1
                    if not index % opt.image_save_sparsity == 0:
                        continue
                    print('sigur ', index)
                    batch_size = val_lr.size(0)
                    valing_results['batch_sizes'] += batch_size
                    lr = val_lr
                    hr = val_hr
                    if torch.cuda.is_available():
                        lr = lr.cuda()
                        hr = hr.cuda()
                    sr = netG(lr)
            
                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                    valing_results['ssims'] += batch_ssim * batch_size
                    valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                    valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                    val_bar.set_description(
                        desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                            valing_results['psnr'], valing_results['ssim']))
            
                    val_images.extend(
                        [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                        display_transform()(sr.data.cpu().squeeze(0))])
                    index += 1
                    print('val imgs len ', len(val_images))
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 15)
                val_save_bar = tqdm(val_images, desc='[saving training results]')
                print('imgs len ', len(val_images))
                index = 0
                for image in val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                    print('saved at ', out_path + 'epoch_%d_index_%d.png' % (epoch, index))
                    index += 1
    
        if epoch % opt.save_params_at_epoch == 0:
            # save model parameters
            torch.save(netG.state_dict(), 'saved_nets/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
            torch.save(netD.state_dict(), 'saved_nets/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
            # save loss\scores\psnr\ssim
            results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
            results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
            results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
            results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
            if not valing_results == None:
                results['psnr'].append(valing_results['psnr'])
                results['ssim'].append(valing_results['ssim'])
    
        # if epoch % 10 == 0 and epoch != 0:
        #     out_path = 'statistics/'
        #     data_frame = pd.DataFrame(
        #         data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
        #               'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
        #         index=range(1, epoch + 1))
        #     data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')