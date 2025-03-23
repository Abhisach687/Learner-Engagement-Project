#!/usr/bin/env python
"""
Filename: mini_lmdb_time_saver_single_thread.py

Key Features:
  - Partial sampling (SAMPLE_RATIO) to only process a fraction of CSV lines
  - Batch-based GPU extraction (BATCH_SIZE)
  - Skips flow if resolution < SKIP_FLOW_BELOW
  - Single-transaction LMDB writes (no parallel writes -> won't hang on Windows)
  - Logs how many items are written & total time, so you can see speed-up

Usage:
  1) Adjust top-level variables (CSV_PATHS, RESOLUTIONS, SAMPLE_RATIO, etc.)
  2) Run: python mini_lmdb_time_saver_single_thread.py
  3) Confirm it processes some frames and doesn't get stuck
  4) See console logs for how many items got written & time taken

Then apply the same logic to your main, full LMDB generation code.

"""

import os
import time
import random
import pickle
import lmdb
import torch
import timm
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch import autocast

# -----------------------------
# USER CONFIG TOGGLES
# -----------------------------
SAMPLE_RATIO      = 0.05     # fraction of lines to process in CSV for quick test
BATCH_SIZE        = 16       # GPU batch size for frames
NUM_FRAMES        = 50       # frames per video
SKIP_FLOW_BELOW   = 300      # if is_flow & res<300 => skip
LMDB_MAP_SIZE_GB  = 1        # 1GB for test
LMDB_MAP_SIZE     = LMDB_MAP_SIZE_GB * (1024**3)
RESOLUTIONS       = [112, 224, 300]  # we test these res for the mini-run

# CSV / directory structure
BASE_DIR      = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project")
DATA_DIR      = BASE_DIR / "data" / "DAiSEE"
FRAMES_DIR    = DATA_DIR / "ExtractedFrames"
CSV_DIR       = DATA_DIR / "Labels"
CSV_PATHS     = [
    CSV_DIR / "TrainLabels.csv",
    CSV_DIR / "ValidationLabels.csv",
    CSV_DIR / "TestLabels.csv"
]

# Where to store LMDB
LMDB_OUTPUT_DIR = BASE_DIR / "mini_lmdb_singthread_results"
LMDB_OUTPUT_DIR.mkdir(exist_ok=True)

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
TRANSFORM_RGB = transforms.Compose([
    transforms.Resize((300, 300)),  # always do 300 so we can keep consistent in this mini-run
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
TRANSFORM_FLOW = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])  # example if single-channel
])

print(f"\n=== mini_lmdb_time_saver_single_thread.py ===")
print(f"DEVICE={device}, SAMPLE_RATIO={SAMPLE_RATIO}, BATCH_SIZE={BATCH_SIZE}")
print(f"SKIP_FLOW_BELOW={SKIP_FLOW_BELOW}, LMDB_MAP_SIZE={LMDB_MAP_SIZE_GB}GB\n")

def partial_sample_lines(lines, ratio):
    """Return only ratio fraction of CSV lines for quick test."""
    out = []
    for ln in lines:
        if random.random() < ratio:
            out.append(ln)
    return out

def uniform_frame_select(folder: Path, num_frames=50):
    """Uniformly pick `num_frames` from sorted frames. If folder has <= num_frames, return them all."""
    frames = sorted(folder.glob("frame_*.jpg"))
    if len(frames) <= num_frames:
        return frames
    idxes = np.linspace(0, len(frames)-1, num_frames, dtype=int)
    return [frames[i] for i in idxes]

def guess_split_subfolder(csv_stem: str):
    """If 'TrainLabels' => 'Train', 'Validation' => 'Validation', 'Test' => 'Test'."""
    stem_lower = csv_stem.lower()
    if "train" in stem_lower:
        return "Train"
    elif "val" in stem_lower:
        return "Validation"
    elif "test" in stem_lower:
        return "Test"
    return "Train"  # fallback

def load_image(fp: Path, is_flow: bool):
    """Load & transform one frame. If fails, return a blank fallback."""
    try:
        img = Image.open(fp)
        if is_flow:
            img = img.convert("L")
            return TRANSFORM_FLOW(img).unsqueeze(0) # shape: (1,1,300,300)
        else:
            img = img.convert("RGB")
            return TRANSFORM_RGB(img).unsqueeze(0)  # shape: (1,3,300,300)
    except:
        # fallback
        if is_flow:
            blank = Image.new("L",(300,300))
            return TRANSFORM_FLOW(blank).unsqueeze(0)
        else:
            blank = Image.new("RGB",(300,300))
            return TRANSFORM_RGB(blank).unsqueeze(0)

def build_lmdb(csv_file: Path, resolution:int, is_flow:bool):
    """
    1) Skip if (is_flow & resolution<SKIP_FLOW_BELOW)
    2) partial sample lines from CSV
    3) subfolder => 'Train'|'Validation'|'Test' based on CSV name
    4) do uniform_frame_select => up to 50 frames
    5) batch-based GPU forward pass
    6) single-transaction writes in LMDB (no parallel => won't hang)
    """
    if is_flow and resolution < SKIP_FLOW_BELOW:
        print(f"[SKIP] flow < {SKIP_FLOW_BELOW}px => {csv_file.name} {resolution}px")
        return 0

    if not csv_file.exists():
        print(f"[WARN] => CSV not found => {csv_file}")
        return 0
    lines_raw = csv_file.read_text().strip().splitlines()
    if len(lines_raw)<=1:
        print(f"[WARN] => CSV empty => {csv_file}")
        return 0
    header = lines_raw[0].split(",")
    try:
        clip_idx = header.index("ClipID")
    except:
        clip_idx=0

    # partial sample
    lines = lines_raw[1:]
    lines_sampled = partial_sample_lines(lines, SAMPLE_RATIO)
    if not lines_sampled:
        print(f"[INFO] => 0 lines after sampling => {csv_file.name}")
        return 0

    # prep backbone
    in_ch = 1 if is_flow else 3
    backbone = timm.create_model("tf_efficientnetv2_l", pretrained=True, in_chans=in_ch)
    backbone.reset_classifier(0)
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad=False

    # LMDB path
    tag = "flow" if is_flow else "rgb"
    lmdb_name = f"{csv_file.stem}_{tag}_{resolution}.mdb"
    lmdb_path = LMDB_OUTPUT_DIR / lmdb_name
    if lmdb_path.exists():
        # remove old if any
        for ff in lmdb_path.glob("*"):
            ff.unlink()
        lmdb_path.rmdir()
    env = lmdb.open(str(lmdb_path), map_size=LMDB_MAP_SIZE)

    t0 = time.time()
    written_count=0
    with torch.no_grad(), env.begin(write=True) as txn:
        pbar = tqdm(lines_sampled, desc=f"LMDB {csv_file.stem} {tag}_{resolution}")
        for row_str in pbar:
            row_split = row_str.strip().split(",")
            clip_id   = row_split[clip_idx].split(".")[0]
            subfolder = guess_split_subfolder(csv_file.stem)
            video_folder = FRAMES_DIR / subfolder / clip_id
            if not video_folder.exists():
                continue
            frames = uniform_frame_select(video_folder, NUM_FRAMES)
            if not frames:
                continue

            feats_list=[]
            nbatches = (len(frames) + BATCH_SIZE -1)//BATCH_SIZE
            for b_idx in range(nbatches):
                b_fps = frames[b_idx*BATCH_SIZE:(b_idx+1)*BATCH_SIZE]
                images=[]
                for fp_ in b_fps:
                    images.append(load_image(fp_, is_flow))
                batch_tensor = torch.cat(images, dim=0).to(device) # shape (B, C, 300,300)
                with autocast(device_type='cuda',dtype=torch.float16):
                    out = backbone(batch_tensor)  # shape (B, feat_dim)
                out_np = out.cpu().half().numpy()
                for row_ in out_np:
                    feats_list.append(row_)

            feats_array = np.stack(feats_list, axis=0) # shape (T, feat_dim)
            # store
            key_str= f"{subfolder}_{clip_id}_{tag}_{resolution}".encode("utf-8")
            val_bytes= pickle.dumps(feats_array)
            txn.put(key_str, val_bytes)
            written_count+=1

    elapsed = time.time()-t0
    print(f"[DONE] => wrote {written_count} items => {lmdb_name} in {elapsed:.2f}s")
    env.close()
    return written_count

def main():
    start_all= time.time()
    print("[mini_lmdb_time_saver_single_thread] => START")
    print(f"SAMPLE_RATIO={SAMPLE_RATIO}, BATCH_SIZE={BATCH_SIZE}, SKIP_FLOW_BELOW={SKIP_FLOW_BELOW}")
    print(f"LMDB_OUTPUT_DIR => {LMDB_OUTPUT_DIR}\n")

    total_items=0
    for csvf in CSV_PATHS:
        for res in RESOLUTIONS:
            # rgb
            c_rgb= build_lmdb(csvf, res, is_flow=False)
            total_items+= c_rgb
            # flow
            c_flow= build_lmdb(csvf, res, is_flow=True)
            total_items+= c_flow

    total_time = time.time()-start_all
    print(f"\n[ALL DONE] => total LMDB items={total_items}, total time={total_time:.2f}s\n")
    print("Check the logs above to see how many frames were processed at each resolution,")
    print("and how long each pass took. Then choose which time-saving features to")
    print("apply in your full LMDB generation script (batch extraction, partial sampling, skipping flow, etc.).")

if __name__=="__main__":
    random.seed(42)  # for consistent sampling
    main()
