import os
import sys
sys.path.append(os.path.abspath(os.path.join("/home/hds3304/Desktop/RBE595/P4/Code/", '../external/RAFT/core')))
import argparse
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, save_path, idx):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    #cv2.waitKey()
    img_flo_bgr = cv2.cvtColor(img_flo, cv2.COLOR_RGB2BGR)

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f'flow_{idx:05d}.png')
    cv2.imwrite(out_file, img_flo_bgr)
    print(f"Saved {out_file}")

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        images = sorted(images)

        save_folder = os.path.join("/home/hds3304/Desktop/RBE595/Navigating_through_unknown-Vizflyt/output", "flow_output")
        os.makedirs(save_folder, exist_ok=True)

        for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, save_folder, idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
    
#python raft_of.py --model=/home/hds3304/Desktop/RBE595/P4/RAFT_Models/raft-things.pth --path=/home/hds3304/Desktop/RBE595/P4/data/p4_colmap_nov6_1000/images_4
