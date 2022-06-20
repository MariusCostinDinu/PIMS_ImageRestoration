import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from argparse import ArgumentParser
import os
import time

from inpainting_data import dataset
from inpainting_utils.tools import add_padding_to_all_sides, get_config, random_bbox, mask_image

from trainer import Trainer
from super_resolution_model.super_resolution_model import SR_Generator

import custom_resize

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='inpainting_configs/config.yaml',
                    help="training configuration")

config = None

def resize_and_save_images(x, inpainted_result, offset_flow, iteration, config):
    # x shape is [B, C, H, W] 
    res_size = config['size_for_super_res_input']

    x_resized = torch.zeros(0, x.shape[1], res_size, res_size)
    inpainted_result_resized = torch.zeros(0, x.shape[1], res_size, res_size)
    offset_flow_resized = torch.zeros(0, x.shape[1], res_size, res_size)

    for (single_image_x, single_image_inpaint, single_image_flow) in zip(x, inpainted_result, offset_flow):
        x_resized = torch.cat((x_resized, \
            custom_resize.my_resize(single_image_x, 200).view(1, x.shape[1], res_size, res_size)))
        inpainted_result_resized = torch.cat((inpainted_result_resized, \
            custom_resize.my_resize(single_image_inpaint, 200).view(1, x.shape[1], res_size, res_size)))
        offset_flow_resized = torch.cat((offset_flow_resized, \
            custom_resize.my_resize(single_image_flow, 200).view(1, x.shape[1], res_size, res_size)))

    viz_images = torch.stack([x_resized, inpainted_result_resized, offset_flow_resized], dim=1)
    viz_images = viz_images.view(-1, *list(x_resized.size())[1:])
    vutils.save_image(viz_images, # image to be saved
                    '%s/niter_%03d.png' % ('output_after_inpainting/r1/r2', iteration), # path where the image is saved
                    nrow=3 * 4,
                    normalize=True)

    return x_resized, inpainted_result_resized

def main():
    torch.cuda.empty_cache()

    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in config['gpu_ids'])
        cudnn.benchmark = True

    device = torch.device("cuda:0") if cuda else torch.device('cpu')

    try:
        # Load the dataset
        test_inpainting_dataset = dataset.InpaintingDataset(data_path=config['input_images_path'],
                                            image_shape=config['image_shape'],              # [256,256,3]]
                                            with_subfolder=config['data_with_subfolder'],   # True
                                            random_crop=config['random_crop'])              # True
        test_inpainting_loader = torch.utils.data.DataLoader(dataset=test_inpainting_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=config['num_workers'])

        # Define the trainer
        inpainting_inference = Trainer(config)

        # Get the resume iteration to restart training
        iterable_test_loader = iter(test_inpainting_loader)

        # Load parameters from .pt file
        loaded_gen_model = torch.load(os.path.join(config['inpainting_model_params'], 'gen_%08d.pt' % config['resume_iteration']))
        loaded_weights = {}
        for param_name, param in loaded_gen_model.items():
            loaded_weights['netG.%s' % param_name] = param.to(device)
        inpainting_inference.load_state_dict(loaded_weights, strict=False)

        # Define the super resolution Generator and load parameters for it:
        super_res_inference = SR_Generator(config['super_res_upscale_factor'])
        super_res_inference.eval()

        loaded_netG_weights = torch.load('super_resolution_model_params/netG_epoch_%d_%d.pth' % (config['super_res_upscale_factor'], config['resume_iteration'] * 0 + 20))
        netG_weights = {}
        for k, v in loaded_netG_weights.items():
            netG_weights[k] = v.to(device)
        super_res_inference.load_state_dict(netG_weights)

        # The pipeline of operations (inpaint | super-resolution)
        for iteration in range(1, test_inpainting_dataset.__len__() + 1):
            ground_truth = next(iterable_test_loader)

            # inpainting inference:
            bboxes = random_bbox(config, batch_size=ground_truth.size(0), fixed_position=True)
            x, mask = mask_image(ground_truth, bboxes, config)

            if cuda:
                x = x.cuda()
                mask = mask.cuda()
                ground_truth = ground_truth.cuda()

            inpainted_result, offset_flow = inpainting_inference(x, mask)
            x_resized, inpainted_result_resized = resize_and_save_images(x, inpainted_result, offset_flow, iteration, config)

            # super-resolution inference:
            high_res_image = super_res_inference(inpainted_result_resized)

            padding_size = (config['size_for_super_res_input'] * config['super_res_upscale_factor'] - \
                config['size_for_super_res_input']) / 2.
            x_resized_padded = add_padding_to_all_sides(x_resized, padding_size)
            inpainted_result_resized_padded = add_padding_to_all_sides(inpainted_result_resized, padding_size)

            # saving the final result of the pipeline
            viz_images = torch.stack([x_resized_padded, inpainted_result_resized_padded, high_res_image], dim=1)
            viz_images = viz_images.view(-1, *list(x_resized_padded.size())[1:])
            vutils.save_image(viz_images, # image to be saved
                        '%s/niter_%03d.png' % ('output_after_super_resolution/r1/r2', iteration), # path where the image is saved
                        nrow=3 * 4,
                        normalize=True)


    except Exception as e:
        print('ERROR {}'.format(e))
        raise e

if __name__ == '__main__':
    main()