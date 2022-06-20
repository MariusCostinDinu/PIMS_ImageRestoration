import imageio
import numpy as np
from argparse import ArgumentParser

import torch

from trainer import Trainer
from utils.tools import get_config
import torchvision.utils as vutils
from data.dataset import Dataset

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output3.png', type=str,
                    help='Where to write output.')
parser.add_argument('--model-path', default='', type=str,
                    help='Path to save model')
args = parser.parse_args()


# EXAMPLE: RUN THIS FILE (IN THIS DIRECTORY) USING: python test_tf_model.py --image="input_256x256.jpg" --mask="mask_256x256.jpg" --model-path="torch_model.p"


def main():
    config = get_config(args.config)
    if config['cuda']:
        device = torch.device("cuda:{}".format(config['gpu_ids'][0]))
    else:
        device = torch.device("cpu")


    train_dataset = Dataset(data_path=config['train_data_path'],
                                image_shape=config['image_shape'],
                                with_subfolder=config['data_with_subfolder'],
                                random_crop=config['random_crop'])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=True,
                                                num_workers=config['num_workers'])
    
    trainer = Trainer(config)
    # weights = load_weights(args.model_path, device)
    # for k, v in weights.items():
    #     print(k, ': ', v.shape)
    #trainer.load_state_dict(weights, strict=False)
    #trainer.load_state_dict(torch.load('checkpoints/imagenet/hole_benchmark/gen_00001000.pt'))
    my_weights = load_weights('checkpoints/imagenet/hole_benchmark/gen_00001000.pt', device)
    my_weights_netg = {}
    for k, v in my_weights.items():
        my_weights_netg['netG.%s' % k] = v
    for k, v in my_weights_netg.items():
        print(k, ': ', v.shape)
    trainer.load_state_dict(my_weights_netg, strict=False)
    trainer.eval()

    iterable_train_loader = iter(train_loader)
    ground_truth = next(iterable_train_loader).cuda()   # [B, C, H, W]      # values between [-1, 1]
    print('gt 0 ', ground_truth[0])

    image = imageio.imread(args.image)  # [H, W, C]     # values between [0, 255]
    print('img ', image)
    print('IMAGE ', image.shape)
    print('GT ', ground_truth.shape)
    image = image[:, :, :3]
    print('mg shape', image.shape)
    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).cuda()   # [1, C, H, W]
    image = ground_truth[:1, :, :, :] + torch.ones_like(ground_truth[:1, :, :, :])
    image = image * 127.5
    mask = imageio.imread(args.mask)    # [H, W, C]
    print('mask before ', mask.shape)
    mask = (torch.FloatTensor(mask[:, :, 0]) / 255).unsqueeze(0).unsqueeze(0).cuda()    # [1, 1, H, W]

    x = (image / 127.5 - 1) * (1 - mask).cuda() # [1, C, H, W]      # values between [-1, 1]; values of 0 inside the mask
    with torch.no_grad():
        _, result, _ = trainer.netG(x, mask)    # [1, C, H, W]      # values between [-1, 1]; most values near 0 (like between [-0.1, 0.1]), and very very few ones near -1 or 1

    vutils.save_image(x, 'input_x_tf.png', nrow=3 * 4, normalize=True)
    vutils.save_image(result, 'output2.png', nrow=3 * 4, normalize=True)
    vutils.save_image(ground_truth[:1, :, :, :] + torch.ones_like(ground_truth[:1, :, :, :]), 'from_input.png', nrow=3 * 4, normalize=True)
    vutils.save_image(ground_truth[:1, :, :, :], 'from_input_lalala.png', nrow=3 * 4, normalize=True)
    imageio.imwrite('output3.png', upcast(result[0].permute(1, 2, 0).detach().cpu().numpy()))
    imageio.imwrite('output4.png', (result[0] * 100).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
    imageio.imwrite('output5.png', (result[0] * 1).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))      # THIS IS THE IRRITATING OUTPUT I WAS RECEIVING
    vutils.save_image(result, 'output_vutils.png', nrow=3 * 4, normalize=True)


def load_weights(path, device):
    model_weights = torch.load(path)
    return {
        k: v.to(device)
        for k, v in model_weights.items()
    }


def upcast(x):
    return np.clip((x + 1) * 127.5 , 0, 255).astype(np.uint8)


if __name__ == '__main__':
    main()
