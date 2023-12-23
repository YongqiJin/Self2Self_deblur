import numpy as np
import torch
import torch.nn.functional as F
import network.Punet
import skimage.metrics
from argparse import ArgumentParser
from tqdm import tqdm

import util
import cv2
import os

def data_arg(x, is_flip_lr, is_flip_ud):
    if is_flip_lr > 0:
        x = torch.flip(x, dims=[2])
    if is_flip_ud > 0:
        x = torch.flip(x, dims=[3])
    return x

def get_output(processed, model, drop_rate=0.3, bs=1, device='cpu'):
    processed_tensor = torch.tensor(processed).permute(0,3,1,2).to(device)
    is_flip_lr = np.random.randint(2)
    is_flip_ud = np.random.randint(2)
    processed_tensor = data_arg(processed_tensor, is_flip_lr, is_flip_ud)
    # mask_tensor = torch.ones([bs, model.width, model.height]).to(device)
    mask_tensor = torch.ones(processed_tensor.shape).to(device)
    mask_tensor = F.dropout(mask_tensor, drop_rate) * (1-drop_rate)
    input_tensor = processed_tensor * mask_tensor#.unsqueeze(1)
    output = model(input_tensor, mask_tensor)
    output = data_arg(output, is_flip_lr, is_flip_ud)
    output_numpy = output.detach().cpu().numpy().transpose(0,2,3,1)
    return output_numpy

def get_loss(noisy_blurred, kernel, model, drop_rate=0.3, bs=1, device='cpu'):
    noisy_blurred_tensor = torch.tensor(noisy_blurred).permute(0,3,1,2).to(device)
    is_flip_lr = np.random.randint(2)
    is_flip_ud = np.random.randint(2)
    noisy_blurred_tensor = data_arg(noisy_blurred_tensor, is_flip_lr, is_flip_ud)
    # mask_tensor = torch.ones([bs, model.width, model.height]).to(device)
    mask_tensor = torch.ones(noisy_blurred_tensor.shape).to(device)
    mask_tensor = F.dropout(mask_tensor, drop_rate) * (1-drop_rate)
    input_tensor = noisy_blurred_tensor * mask_tensor#.unsqueeze(1)
    output = model(input_tensor, mask_tensor)
    observe_tensor = 1.0 - mask_tensor#.unsqueeze(1)
    # convolve
    blurred_output = util.convolve_torch(output, kernel.view(1, 1, *kernel.size()), mode='wrap')
    loss = torch.sum((blurred_output-noisy_blurred_tensor).pow(2)*(observe_tensor)) / torch.count_nonzero(observe_tensor).float()
    return loss

def train(file_path, args):
    print(file_path)
    gt = util.load_np_image(file_path)
    _, w, h, c = gt.shape
    model_path = file_path[0:file_path.rfind(".")] + '/' + args.model_type + '/' + '_'.join([str(args.kernel_size), str(args.kernel_sigma), str(args.sigma), str(args.drop_rate)]) + "/"
    os.makedirs(model_path, exist_ok=True)
    kernel = util.generate_kernel(args.kernel_size, 'guassian', args.kernel_sigma)
    blurred = util.blur(gt, model_path, kernel, bs=args.bs)
    noisy_blurred = util.add_gaussian_noise(blurred, model_path, args.sigma, bs=args.bs)
    print('noisy_blurred shape:', noisy_blurred.shape)
    print('image shape:', gt.shape)
    
    # origin
    PSNR = skimage.metrics.peak_signal_noise_ratio(np.squeeze(gt[0]), np.squeeze(noisy_blurred[0]))
    print("Origin psnr = ", PSNR)
    with open(args.log_path, 'a') as f:
        f.write("File path: %s\n" % file_path)
        f.write("Origin psnr  is {:.4f}\n".format(PSNR))
    
    # model
    model = network.Punet.Punet(channel=c, width=w, height=h, drop_rate=args.drop_rate).to(args.device)
    model.train()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            
    # begin training
    avg_loss = 0
    for step in tqdm(range(args.iteration)):
        # one step
        model.train()
        loss = get_loss(noisy_blurred, kernel, model, drop_rate=args.drop_rate, bs=args.bs, device=args.device)
        avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # clear cache
        # with torch.cuda.device(args.device):
        #     torch.cuda.empty_cache()
        
        # test
        if (step+1) % args.test_frequency == 0:
            model.eval()
            print("After %d training step(s)" % (step + 1),
                  "loss  is {:.9f}".format(avg_loss / args.test_frequency))
            final_image = np.zeros(gt.shape)
            for j in tqdm(range(args.num_prediction)):
                output_numpy = get_output(noisy_blurred, model, drop_rate=args.drop_rate, bs=args.bs, device=args.device)
                final_image += output_numpy
                # clear cache
                # with torch.cuda.device(args.device):
                #     torch.cuda.empty_cache()
            final_image = np.squeeze(np.uint8(np.clip(final_image / args.num_prediction, 0, 1) * 255))
            cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '.png', final_image)
            PSNR = skimage.metrics.peak_signal_noise_ratio(np.squeeze(gt[0]), final_image.astype(np.float32)/255.0)
            print("psnr = ", PSNR)
            with open(args.log_path, 'a') as f:
                f.write("After %d training step(s), " % (step + 1))
                f.write("loss  is {:.9f}, ".format(avg_loss / args.test_frequency)) 
                f.write("psnr  is {:.4f}".format(PSNR))
                f.write("\n")
            avg_loss = 0
    
    return PSNR
    
def main(args):
    path = args.path
    file_list = os.listdir(path)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        f.write("Self2self algorithm!\n")
    avg_psnr = 0
    count = 0
    for file_name in file_list:
        if not os.path.isdir(path + file_name):
            PSNR = train(path+file_name, args)
            avg_psnr += PSNR
            count += 1
    with open(args.log_path, 'a') as f:
        f.write('average psnr is {:.4f}'.format(avg_psnr/count))
    util.plot_log(args.log_path)

def build_args():
    parser = ArgumentParser()
    
    parser.add_argument("--iteration", type=int, default=150000)
    parser.add_argument("--test_frequency", type=int, default=1000)
    parser.add_argument("--drop_rate", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--kernel_size", type=int, default=15)
    parser.add_argument("--kernel_sigma", type=float, default=1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--model_type", type=str, default='deblur')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_prediction", type=int, default=100)
    parser.add_argument("--log_path", type=str, default='./logs/log.txt')
    parser.add_argument("--path", type=str, default='./testsets/barbara/')
    parser.add_argument("--device", type=str, default='cpu')
    
    args = parser.parse_args()
    return args
            
if __name__ == "__main__":
    args = build_args()
    main(args)