import sys
import os
sys.path.append(os.path.abspath(os.path.join("/home/hds3304/Desktop/RBE595/P4/Code/", '../external/RAFT/core')))
import cv2
import torch
import numpy as np
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_frame(frame, half=False):
    img = torch.from_numpy(frame).permute(2, 0, 1).float()
    tensor = img[None].to(DEVICE)
    if half:
        tensor = tensor.half()
    return tensor


def flow_only_visualization(flo):
    flo_np = flo[0].permute(1, 2, 0).cpu().numpy()
    flow_img = flow_viz.flow_to_image(flo_np)
    flow_bgr = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)
    return flow_bgr


def remove_module_prefix(state_dict):
    """Strip 'module.' from keys if checkpoint was saved with DataParallel"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def process_video(args, resize_width=384, resize_height=256):
    print(f"Running on device: {DEVICE}")

    # Load RAFT model
    model = RAFT(args)
    checkpoint = torch.load(args.model, map_location=DEVICE)
    checkpoint = remove_module_prefix(checkpoint)
    model.load_state_dict(checkpoint)
    model = model.module if hasattr(model, 'module') else model
    model.to(DEVICE)
    if args.mixed_precision:
        model.half()
    model.eval()

    # Open input video
    if not os.path.isfile(args.video):
        print(f"Video file not found: {args.video}")
        return
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = resize_width
    h = resize_height

    os.makedirs("/home/hds3304/Desktop/RBE595/P4/output", exist_ok=True)
    flow_out_path = os.path.join("/home/hds3304/Desktop/RBE595/P4/output", 'flow_sintel.mp4')
    flow_out = cv2.VideoWriter(flow_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Video is empty.")
        return
    prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    prev_frame_rgb = cv2.resize(prev_frame_rgb, (w, h))

    frame_idx = 0
    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        next_frame_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
        next_frame_rgb = cv2.resize(next_frame_rgb, (w, h))

        # Convert to tensors
        prev_tensor = load_frame(prev_frame_rgb, half=args.mixed_precision)
        next_tensor = load_frame(next_frame_rgb, half=args.mixed_precision)

        # Pad frames
        padder = InputPadder(prev_tensor.shape)
        prev_tensor, next_tensor = padder.pad(prev_tensor, next_tensor)

        # Compute optical flow
        with torch.no_grad():
            _, flow_up = model(prev_tensor, next_tensor, iters=20, test_mode=True)

        flow_bgr = flow_only_visualization(flow_up)
        flow_out.write(flow_bgr)

        prev_frame_rgb = next_frame_rgb
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    flow_out.release()
    print(f"Saved flow-only video: {flow_out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Path to RAFT checkpoint (small model recommended)")
    parser.add_argument('--video', required=True, help="Input video file (MP4)")
    parser.add_argument('--small', action='store_true', help='Use small RAFT model')
    parser.add_argument('--mixed_precision', action='store_true', help='Use float16 to save memory')
    args = parser.parse_args()

    process_video(args)

#python raft_flow_only.py --model /home/hds3304/Desktop/RBE595/P4/RAFT/raft-things-small.pth --video /home/hds3304/Desktop/RBE595/P4/Videos/optical_flow_test.mp4
