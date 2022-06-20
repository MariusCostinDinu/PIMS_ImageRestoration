from configparser import Interpolation
import os
import random
import threading
import time
import shutil
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from PIL import Image
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize
from tensorboardX import SummaryWriter

from trainer import Trainer
from data.dataset import Dataset
from utils.tools import get_config, random_bbox, mask_image
from utils.logger import get_logger

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--model-path', default='', type=str,
                    help='Path to save model')



def generate_ratio(value_1, value_2, inbetween_value):
    return (inbetween_value - value_1) / (value_2 - value_1)

def linear_interp(p1, p2, ratio):
    return p1 + (p2 - p1) * ratio

def bilinear_interp(p11, p12, p21, p22, ratio_y, ratio_x):
    return linear_interp(linear_interp(p11, p12, ratio_x), linear_interp(p21, p22, ratio_x), ratio_y)



def my_resize(img_tensor, new_dim):
    if not(new_dim == img_tensor.shape[1]):
        return resize_by_biliniar_interpolation(img_tensor, new_dim)
    return img_tensor

# resizing to (new_dim, new_dim), where img_tensor is of shape [nb_channels, rows, cols]:
def resize_by_biliniar_interpolation(img_tensor, new_dim):
    old_size = img_tensor.shape # [0: channels_number, 1: lines_number, 2: columns_number]
    new_img_tensor = torch.zeros(old_size[0], new_dim, new_dim)

    img_tensor = img_tensor.permute(1, 2, 0)
    new_img_tensor = new_img_tensor.permute(1, 2, 0)

    for idx_y in range(new_dim):
        for idx_x in range(new_dim):
            value = None
            interp_transpose_y = (idx_y * old_size[1]) / new_dim
            interp_transpose_x = (idx_x * old_size[2]) / new_dim
            interp_idx_y = (idx_y * old_size[1]) // new_dim
            interp_idx_x = (idx_x * old_size[2]) // new_dim
            ratio_y = generate_ratio(interp_idx_y, interp_idx_y + 1, interp_transpose_y)
            ratio_x = generate_ratio(interp_idx_x, interp_idx_x + 1, interp_transpose_x)
            next_interp_idx_y = min(interp_idx_y + 1, old_size[2] - 1)
            next_interp_idx_x = min(interp_idx_x + 1, old_size[1] - 1)

            if (idx_y * old_size[1]) % new_dim == 0:
                if (idx_x * old_size[2]) % new_dim == 0:
                    value = img_tensor[interp_idx_y][interp_idx_x]
                else:
                    value = linear_interp( \
                        img_tensor[interp_idx_y][interp_idx_x], \
                        img_tensor[interp_idx_y][next_interp_idx_x], ratio_x)
            else:
                if (idx_x * old_size[2]) % new_dim == 0:
                    value = linear_interp( \
                        img_tensor[interp_idx_y][interp_idx_x], \
                        img_tensor[next_interp_idx_y][interp_idx_x], ratio_y)
                else:
                    value = bilinear_interp( \
                        img_tensor[interp_idx_y][interp_idx_x], \
                        img_tensor[interp_idx_y][next_interp_idx_x], \
                            img_tensor[next_interp_idx_y][interp_idx_x], \
                                img_tensor[next_interp_idx_y][next_interp_idx_x], ratio_y, ratio_x)

            new_img_tensor[idx_y][idx_x] = value

    return new_img_tensor.permute(2, 0, 1)


config = None
cuda = False
checkpoint_path = None

def resize_and_save_images(x, inpainted_result, offset_flow, iteration):
    global config
    global checkpoint_path

    # x shape is [B, C, H, W] 
    res_size = config['size_for_srgan_input']

    x_resized = torch.zeros(0, x.shape[1], res_size, res_size)
    inpainted_result_resized = torch.zeros(0, x.shape[1], res_size, res_size)
    offset_flow_resized = torch.zeros(0, x.shape[1], res_size, res_size)

    for (single_image_x, single_image_inpaint, single_image_flow) in zip(x, inpainted_result, offset_flow):
        x_resized = torch.cat((x_resized, \
            my_resize(single_image_x, 200).view(1, x.shape[1], res_size, res_size)))
        inpainted_result_resized = torch.cat((inpainted_result_resized, \
            my_resize(single_image_inpaint, 200).view(1, x.shape[1], res_size, res_size)))
        offset_flow_resized = torch.cat((offset_flow_resized, \
            my_resize(single_image_flow, 200).view(1, x.shape[1], res_size, res_size)))

    viz_max_out = config['viz_max_out']
    if x.size(0) > viz_max_out:
        viz_images = torch.stack([x_resized[:viz_max_out], inpainted_result_resized[:viz_max_out],
                                offset_flow_resized[:viz_max_out]], dim=1)
    else:
        viz_images = torch.stack([x_resized, inpainted_result_resized, offset_flow_resized], dim=1)
    viz_images = viz_images.view(-1, *list(x_resized.size())[1:])
    vutils.save_image(viz_images, # image to be saved
                    '%s/niter_%03d.png' % (checkpoint_path, iteration), # path where the image is saved
                    nrow=3 * 4,
                    normalize=True)

def main():
    global config
    global cuda
    global checkpoint_path

    save_threads = []

    args = parser.parse_args()
    config = get_config(args.config)

    torch.cuda.empty_cache()

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True
    
    device = torch.device("cuda:0") if cuda else torch.device('cpu')

    # Configure checkpoint path
    checkpoint_path = os.path.join('checkpoints',
                                   config['dataset_name'],
                                   config['mask_type'] + '_' + config['expname'])
    # => checkpoints/imagenet/hole_benchmark
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    # The file 'configs/config.yaml' is copied to a new path (created above): 'checkpoints/imagenet/hole_benchmark/config.yaml'
    writer = SummaryWriter(logdir=checkpoint_path)

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    try:  # for unexpected error logging
        # Load the dataset
        train_dataset = Dataset(data_path=config['train_data_path'],
                                image_shape=config['image_shape'],              # [256,256,3]]
                                with_subfolder=config['data_with_subfolder'],   # True
                                random_crop=config['random_crop'])              # True
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=config['num_workers'])

        # Define the trainer
        trainer = Trainer(config)
        trainer_module = trainer

        # # Get the resume iteration to restart training
        start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1
        iterable_train_loader = iter(train_loader)

        if config['resume_params']:
            loaded_gen_model = torch.load(os.path.join(checkpoint_path, 'gen_%08d.pt' % config['resume_iteration']))
            loaded_dis_model = torch.load(os.path.join(checkpoint_path, 'dis_%08d.pt' % config['resume_iteration']))
            loaded_opt_model = torch.load(os.path.join(checkpoint_path, 'optimizer.pt'))

            loaded_weights = {}

            for param_name, param in loaded_gen_model.items():
                loaded_weights['netG.%s' % param_name] = param.to(device)

            for net_name, net in loaded_dis_model.items():
                for param_name, param in net.items():
                    loaded_weights['%s.%s' % (net_name, param_name)] = param.to(device)

            for net_name, net in loaded_opt_model.items():
                for param_name, param in net.items():
                    loaded_weights['%s.%s' % (net_name, param_name)] = param

            trainer.load_state_dict(loaded_weights, strict=False)

        time_count = time.time()
        iterator_progress = 0

        torch.autograd.set_detect_anomaly(True)
        # the training is done in config['niter'] number of iterations. It does not have to be proportional with the batch size. The training order is from the beginning to the end of the dataset, then starting all over again, and again, and again, ..., until the config['niter'] is reached (it can even be reached when we are in the middle of the dataset)
        for iteration in range(start_iteration, config['niter'] + 1):
            print('iteration : ', iteration)
            if iterator_progress < train_dataset.__len__():
                ground_truth = next(iterable_train_loader)
            else:
                iterator_progress = 0
                iterable_train_loader = iter(train_loader)  # comes back at the beginning of the dataset
                ground_truth = next(iterable_train_loader)

            iterator_progress = iterator_progress + config['batch_size']

            # Prepare the inputs
            bboxes = random_bbox(config, batch_size=ground_truth.size(0))
            x, mask = mask_image(ground_truth, bboxes, config)

            # the mask's dimensions are 128 minus (a random number between 0 and config['max_delta_shape']). That one is like a random padding applied simetrically to a side and the opposed one (on width / on height)
            
            if cuda:
                x = x.cuda()
                mask = mask.cuda()
                ground_truth = ground_truth.cuda()
            
            # x shape: [batch_size, 3, 256, 256], except for the last batch where first dim might be less
            # mask shape: [batch_size, 1, 256, 256]

            ###### Forward pass ######
            compute_g_loss = iteration % config['n_critic'] == 0
            
            losses, inpainted_result, offset_flow = trainer(x, bboxes, mask, ground_truth, compute_g_loss)
            # inpainted_result -- values between [0, 1]

            # gpu_usage('after trainer %s' % iteration)
            # Scalars from different devices are gathered into vectors
            for k in losses.keys():
                if not losses[k].dim() == 0:
                    losses[k] = torch.mean(losses[k])

            ###### Backward pass ######
            # Update D
            trainer_module.optimizer_d.zero_grad()
            losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
            losses['d'].backward()


            # Update G
            if compute_g_loss:
                trainer_module.optimizer_g.zero_grad()
                losses['g'] = losses['l1'] * config['l1_loss_alpha'] \
                              + losses['ae'] * config['ae_loss_alpha'] \
                              + losses['wgan_g'] * config['gan_loss_alpha']
                #losses['g'] = losses['wgan_g'] * config['gan_loss_alpha']
                losses['g'].backward()
                trainer_module.optimizer_g.step()

            trainer_module.optimizer_d.step()


            # Log and visualization
            log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
            if iteration % config['print_iter'] == 0:
                time_count = time.time() - time_count
                speed = config['print_iter'] / time_count
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()

                message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
                for k in log_losses:
                    v = losses.get(k, 0.)
                    writer.add_scalar(k, v, iteration)
                    message += '%s: %.6f ' % (k, v)
                message += speed_msg
                print('INFO: ', message)

            if iteration % (config['viz_iter']) == 0:
                if cuda:
                    x = x.cpu()
                    inpainted_result = inpainted_result.detach().cpu()
                    offset_flow = offset_flow.detach().cpu()

                new_save_thread = threading.Thread(target=resize_and_save_images, args=(x, inpainted_result, offset_flow, iteration), daemon=True)
                new_save_thread.start()
                save_threads.append(new_save_thread)

            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(checkpoint_path, iteration)
        
        for thread_ in save_threads:
            thread_.join()

    except Exception as e:  # for unexpected error logging
        print('ERROR {}'.format(e))
        raise e

if __name__ == '__main__':
    main()
