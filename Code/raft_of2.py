import os
import sys
sys.path.append(os.path.abspath(os.path.join("/home/hds3304/Desktop/RBE595/Navigating_through_unknown-Vizflyt/Code/", '../RAFT/core')))
import argparse
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_frame(frame):
    img = torch.from_numpy(frame).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def generate_visualization(img, flo):
    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    #flo_np = flo[0].permute(1, 2, 0).cpu().numpy()
    #flow_img = flow_viz.flow_to_image(flo_np)

    # Concatenate original + flow vertically
    combined_img = np.concatenate([img_np, flow_img], axis=0)

    # Convert to BGR for OpenCV
    combined_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
    #flow_bgr = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)

    return combined_bgr, #flow_bgr


def process_video(args):
    # Load RAFT model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model = model.module
    model.to(DEVICE)
    model.eval()

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output folder
    save_folder = "/home/hds3304/Desktop/RBE595/Navigating_through_unknown-Vizflyt/output"
    os.makedirs(save_folder, exist_ok=True)

    # Video writers
    combined_out = cv2.VideoWriter(os.path.join(save_folder, 'combined_flow.mp4'),
                                   cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h*2))
    #flow_out = cv2.VideoWriter(os.path.join(save_folder, 'flow_only.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Video is empty.")
        return
    prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)

    frame_idx = 0
    while True:
        ret, next_frame = cap.read()
        if not ret:
            break
        next_frame_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)

        # Convert frames to tensors
        prev_tensor = load_frame(prev_frame_rgb)
        next_tensor = load_frame(next_frame_rgb)

        # Pad frames
        padder = InputPadder(prev_tensor.shape)
        prev_tensor, next_tensor = padder.pad(prev_tensor, next_tensor)

        # Compute optical flow
        with torch.no_grad():
            flow_low, flow_up = model(prev_tensor, next_tensor, iters=20, test_mode=True)

        # Generate visualizations
        combined_bgr, flow_bgr = generate_visualization(prev_tensor, flow_up)

        combined_out.write(combined_bgr)
        flow_out.write(flow_bgr)

        prev_frame_rgb = next_frame_rgb
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    combined_out.release()
    flow_out.release()
    print(f"Saved combined video: {os.path.join(save_folder, 'combined_flow.mp4')}")
    print(f"Saved flow-only video: {os.path.join(save_folder, 'flow_only.mp4')}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Path to RAFT checkpoint")
    parser.add_argument('--video', required=True, help="Input video file (mp4)")
    parser.add_argument('--small', action='store_true', help='Use small RAFT model')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='Use efficient correlation')
    args = parser.parse_args()

    process_video(args)
