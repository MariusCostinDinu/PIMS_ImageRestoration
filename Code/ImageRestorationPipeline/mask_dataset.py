import torch
import torch.backends.cudnn as cudnn
import os
from argparse import ArgumentParser
import torchvision.utils as vutils

from inpainting_data import dataset
from inpainting_utils.tools import add_padding_to_all_sides, get_config, random_bbox, mask_image
import custom_resize



parser = ArgumentParser()
parser.add_argument('--config', type=str, default='inpainting_configs/config.yaml',
                    help="training configuration")

def main():
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
        image_dataset = dataset.InpaintingDataset(data_path=config['input_images_path'],
                                            image_shape=config['image_shape'],              # [256,256,3]]
                                            with_subfolder=config['data_with_subfolder'],   # True
                                            random_crop=config['random_crop'])              # True
        image_loader = torch.utils.data.DataLoader(dataset=image_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=config['num_workers'])

        iterable_test_loader = iter(image_loader)
        for image_number in range(1, image_dataset.__len__() + 1):
            ground_truth = next(iterable_test_loader)

            bboxes = random_bbox(config, batch_size=ground_truth.size(0), fixed_position=True)
            x, _ = mask_image(ground_truth, bboxes, config, fixed_position=True)
            #x = custom_resize.my_resize(x, config['size_for_super_res_input'])
            vutils.save_image(x, '%s/r1/r2/%d.png' % (config['input_masked_images_path'], image_number), padding=0, normalize=True)

    except Exception as e:
        print('ERROR {}'.format(e))
        raise e

    

if __name__ == '__main__':
    main()