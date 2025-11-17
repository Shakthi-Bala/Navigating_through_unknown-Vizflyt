import os
import sys
import cv2
import torch
import numpy as np
import argparse

sys.path.append(os.path.abspath(os.path.join("/home/hds3304/Desktop/RBE595/P4/Code/",'../external/RAFT/core')))

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = "cuda"


def load_frame_cv(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float()
    return t[None].to(DEVICE)


def compute_flow(model, f1, f2):
    padder = InputPadder(f1.shape)
    f1, f2 = padder.pad(f1, f2)
    flow_low, flow_up = model(f1, f2, iters=20, test_mode=True)
    return flow_up


def flow_to_vis(flow):
    flo = flow[0].permute(1,2,0).cpu().numpy()
    vis = flow_viz.flow_to_image(flo)
    return vis


def detect_gaps(flow_vis):
    gray = cv2.cvtColor(flow_vis, cv2.COLOR_RGB2GRAY)
    mag = gray.astype(np.float32)

    mag_thresh = 2.0
    _, gap_mask = cv2.threshold(mag, mag_thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((7,7), np.uint8)
    close = cv2.morphologyEx(gap_mask, cv2.MORPH_CLOSE, kernel)
    open_ = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
    gap_clean = cv2.convertScaleAbs(open_)

    contours, _ = cv2.findContours(gap_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = flow_vis.copy()
    for cnt in contours:
        cv2.drawContours(result, [cnt], -1, (0,0,0), 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
            cv2.circle(result, (cX, cY), 5, (0,0,0), -1)
            cv2.putText(result, "C", (cX-5, cY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return result


def main(video_path, model_path, output_path):
    print("Loading RAFT model...")

    raft_args = argparse.Namespace(
        small=False,
        mixed_precision=False,
        alternate_corr=False
    )

    model = RAFT(raft_args)

    # --- Load checkpoint with module prefix fix ---
    checkpoint = torch.load(model_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        # strip 'module.' prefix if it exists
        if k.startswith("module."):
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

    model = model.to(DEVICE)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading input video.")
        return

    h, w, _ = prev_frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    prev_t = load_frame_cv(prev_frame)
    frame_idx = 0
    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_t = load_frame_cv(frame)
        with torch.no_grad():
            flow = compute_flow(model, prev_t, curr_t)
        flow_vis = flow_to_vis(flow)
        result = detect_gaps(flow_vis)
        out.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        prev_t = curr_t
        frame_idx += 1
        print(f"Frame {frame_idx} processed", end="\r", flush=True)

    cap.release()
    out.release()
    print("\nDone! Output video saved at:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--model", required=True, help="RAFT model .pth")
    parser.add_argument("--output", required=True, help="Output .mp4 file")
    args = parser.parse_args()
    main(args.video, args.model, args.output)

