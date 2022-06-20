import torch

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