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

def process_and_save(image_tensor, flow_tensor, save_path, idx):

    # Convert to numpy
    flow = flow_tensor[0].permute(1,2,0).cpu().numpy()

    # Flow visualization (RGB)
    flow_viz_img = flow_viz.flow_to_image(flow)
    flow_viz_bgr = cv2.cvtColor(flow_viz_img, cv2.COLOR_RGB2BGR)

    # ----------------------------
    # Compute flow magnitude
    # ----------------------------
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    mag = np.sqrt(u*u + v*v).astype(np.float32)

    mag_thresh = 2.0  # low-motion threshold

    _, gap_mask = cv2.threshold(mag, mag_thresh, 255, cv2.THRESH_BINARY_INV)
    gap_mask = cv2.convertScaleAbs(gap_mask)

    gray = cv2.cvtColor(flow_viz_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Morphology
    morph_kernel = np.ones((7,7), np.uint8)
    mag_close = cv2.morphologyEx(gap_mask, cv2.MORPH_CLOSE, morph_kernel)
    mag_clean = cv2.morphologyEx(mag_close, cv2.MORPH_OPEN, morph_kernel)

    # Find contours of low-flow regions
    contours, _ = cv2.findContours(mag_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw on FLOW visualization only
    result = flow_viz_bgr.copy()

    # ----- Draw ALL contours + centroids -----
    if contours:
        for cnt in contours:

            # Draw contour in BLACK
            cv2.drawContours(result, [cnt], -1, (0,0,0), 2)

            # Compute centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw centroid in BLACK
                cv2.circle(result, (cX, cY), 5, (0,0,0), -1)
                cv2.putText(result, "C", (cX - 5, cY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f"gap_detector_{idx:05d}.png")
    cv2.imwrite(out_file, result)

    print(f"[{idx}] Saved:", out_file)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = sorted(glob.glob(os.path.join(args.path, '*.png')) +
                        glob.glob(os.path.join(args.path, '*.jpg')))

        save_folder = os.path.join("/home/hds3304/Desktop/RBE595/P4/output","gap_detector")
        os.makedirs(save_folder, exist_ok=True)

        for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            process_and_save(image1, flow_up, save_folder, idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    args = parser.parse_args()

    demo(args)

