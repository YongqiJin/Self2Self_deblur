import numpy as np
import cv2
import scipy.io as sio
import scipy
import random
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def add_gaussian_noise(img, model_path, sigma, bs=1):
    index = model_path.rfind("/")
    if sigma > 0:
        # noise = np.random.normal(scale=sigma / 255., size=img.shape).astype(np.float32)
        noise = np.random.normal(scale=sigma / 255., size=[bs, img.shape[1], img.shape[2], img.shape[3]]).astype(np.float32)
        sio.savemat(model_path[0:index] + '/noise.mat', {'noise': noise})
        noisy_img = (img + noise).astype(np.float32)
    else:
        noisy_img = img.astype(np.float32)
    cv2.imwrite(model_path[0:index] + '/noisy.png',
                np.squeeze(np.int32(np.clip(noisy_img, 0, 1) * 255.)))
    return noisy_img


def load_np_image(path, is_scale=True):
    img = cv2.imread(path, -1)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    if is_scale:
        img = np.array(img).astype(np.float32) / 255.
    return img


def mask_pixel(img, model_path, rate):
    index = model_path.rfind("/")
    masked_img = img.copy()
    mask = np.ones_like(masked_img)
    perm_idx = [i for i in range(np.shape(img)[1] * np.shape(img)[2])]
    random.shuffle(perm_idx)
    for i in range(np.int32(np.shape(img)[1] * np.shape(img)[2] * rate)):
        x, y = np.divmod(perm_idx[i], np.shape(img)[2])
        masked_img[:, x, y, :] = 0
        mask[:, x, y, :] = 0
    cv2.imwrite(model_path[0:index] + '/masked_img.png', np.squeeze(np.uint8(np.clip(masked_img, 0, 1) * 255.)))
    cv2.imwrite(model_path[0:index] + '/mask.png', np.squeeze(np.uint8(np.clip(mask, 0, 1) * 255.)))
    return masked_img, mask


def blur(img, model_path, kernel, bs=1):
    index = model_path.rfind("/")
    sio.savemat(model_path[0:index] + '/kernel.mat', {'kernel': kernel})
    kernel = kernel.unsqueeze(0).unsqueeze(-1)
    blurred_img = scipy.ndimage.convolve(img, kernel, mode='wrap').astype(np.float32)
    cv2.imwrite(model_path[0:index] + '/blurred.png',
                np.squeeze(np.int32(np.clip(blurred_img, 0, 1) * 255.)))
    return blurred_img


def generate_kernel(kernel_size, type='guassian', kernel_sigma=1):
    if type == 'guassian':
        x, y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size) ,indexing='xy')
        kernel = torch.exp(-((x - kernel_size // 2)**2 + (y - kernel_size // 2)**2) / (2 * kernel_sigma**2))
        kernel /= kernel.sum() # normalization
    else:
        Warning('Unknown kernel type.')
    return kernel

def convolve_torch(img, kernel, mode='wrap'):

    def periodic_padding(tensor, pad):
        upper_pad = tensor[... , -pad:, :]
        lower_pad = tensor[... , :pad, :]
        tensor = torch.cat((upper_pad, tensor, lower_pad), dim=-2)
        left_pad = tensor[... , -pad:]
        right_pad = tensor[... , :pad]
        tensor = torch.cat((left_pad, tensor, right_pad), dim=-1)
        return tensor
    
    pad = kernel.shape[-1] // 2
    if mode == 'wrap':
        img_padded = periodic_padding(img, pad)
    else:
        Warning('Unknown padding mode.')
    conv_result = F.conv2d(img_padded, kernel, padding=0)
    
    return conv_result


def plot_log(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    steps = []
    losses = []
    psnrs = []
    for line in lines:
        if 'Origin' in line:
            origin_psnr = float(line.split('Origin psnr  is ')[1].split('\n')[0])
        if 'After' in line:
            step = int(line.split('After ')[1].split(' training')[0])
            steps.append(step / 1000)
            loss = float(line.split('loss  is ')[1].split(',')[0])
            losses.append(loss)
            psnr = float(line.split('psnr  is ')[1].split(',')[0])
            psnrs.append(psnr)
    # 绘制loss曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses)
    plt.title('Loss over time')
    plt.xlabel('Iteration (x1000)')
    plt.ylabel('Loss')
    # 绘制psnr曲线
    plt.subplot(1, 2, 2)
    plt.plot(steps, psnrs)
    plt.title('PSNR over time (Origin PSNR = %.4f)' % origin_psnr)
    plt.xlabel('Iteration (x1000)')
    plt.ylabel('PSNR')
    plt.tight_layout()
    plt.savefig(path[:-4] + '.png')