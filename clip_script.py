# CLIP Embedding
#
# dependencies:
# pip install git+https://github.com/openai/CLIP.git
# pip install torch torchvision pillow pandas numpy
#
# run:
#   python clip_script.py
#

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image

warnings.filterwarnings('ignore')

BASE = os.path.expanduser('~/mmai/mmai_midterm_report')
PHOTOS_DIR = f'{BASE}/interior_photos'
GSV_DIR = f'{BASE}/street_view'
SAT_DIR = f'{BASE}/satellite'

# load csv
df = pd.read_csv(f'{BASE}/boston_cleaned.csv')
N = len(df)
D = 512
BATCH = 64
out_path = f'{BASE}/clip_embeddings.npz'
tmp_path = f'{BASE}/clip_embeddings.tmp.npz'

# checks for stale .tmp from prev save
if os.path.exists(tmp_path):
    raise RuntimeError(
        "stale .tmp checkpoint file found\n"
        "main .npz file is still intact\n"
        f"delete '{tmp_path}' and re-run."
    )

# load existing embeddings if there 
if os.path.exists(out_path):
    old = np.load(out_path)
    emb_text = old['emb_text']
    emb_photos = old['emb_photos']
    emb_gsv = old['emb_gsv']
    emb_sat = old['emb_sat']
    has_text = old['has_text']
    has_photos = old['has_photos']
    has_gsv = old['has_gsv']
    has_sat = old['has_sat']
    if len(emb_text) < N:
        pad = N - len(emb_text)
        emb_text = np.vstack([emb_text,   np.zeros((pad, D), dtype=np.float32)])
        emb_photos = np.vstack([emb_photos, np.zeros((pad, D), dtype=np.float32)])
        emb_gsv = np.vstack([emb_gsv,    np.zeros((pad, D), dtype=np.float32)])
        emb_sat = np.vstack([emb_sat,    np.zeros((pad, D), dtype=np.float32)])
        has_text = np.concatenate([has_text,   np.zeros(pad, dtype=bool)])
        has_photos = np.concatenate([has_photos, np.zeros(pad, dtype=bool)])
        has_gsv = np.concatenate([has_gsv,    np.zeros(pad, dtype=bool)])
        has_sat = np.concatenate([has_sat,    np.zeros(pad, dtype=bool)])
    print(f"Loaded existing: {has_text.sum()} text, {has_photos.sum()} photos, "
          f"{has_gsv.sum()} gsv, {has_sat.sum()} sat")
else:
    emb_text = np.zeros((N, D), dtype=np.float32)
    emb_photos = np.zeros((N, D), dtype=np.float32)
    emb_gsv = np.zeros((N, D), dtype=np.float32)
    emb_sat = np.zeros((N, D), dtype=np.float32)
    has_text = np.zeros(N, dtype=bool)
    has_photos = np.zeros(N, dtype=bool)
    has_gsv = np.zeros(N, dtype=bool)
    has_sat = np.zeros(N, dtype=bool)
    print("starting fresh")

# load clip
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Device: {device}")

model, preprocess = clip.load('ViT-B/32', device=device)
model.eval()


# helper functions

def checkpoint():
    """
    write to .tmp.npz first, then rename over the real file so if
    the process dies during np.savez, only the .tmp is corrupted
    """
    np.savez(tmp_path,
             emb_text=emb_text, emb_photos=emb_photos,
             emb_gsv=emb_gsv, emb_sat=emb_sat,
             has_text=has_text, has_photos=has_photos,
             has_gsv=has_gsv, has_sat=has_sat)
    os.replace(tmp_path, out_path)
    print("Checkpoint saved.")


def make_progress_tracker(label, total):
    """
    prints rate + ETA every 10 properties.
    """
    start_time = time.time()

    def report(count):
        if count % 10 != 0:
            return
        elapsed = time.time() - start_time
        rate = count / elapsed if elapsed > 0 else 0
        remaining = (total - count) / rate if rate > 0 else float('inf')
        mins, secs = divmod(int(remaining), 60)
        hours, mins = divmod(mins, 60)
        eta_str = (f"{hours}h {mins}m {secs}s" if hours > 0
                   else f"{mins}m {secs}s" if mins > 0
                   else f"{secs}s")
        print(f"[{label}] {count}/{total} — "
              f"{rate:.1f} props/sec — ETA: {eta_str}")
    return report


def encode_images_batched(paths, context=""):
    """
    encode a list of image paths with CLIP
    - missing files warnings indicate a data problem.
    - corrupt/unreadable files are logged and skipped.
    - returns float32 array of shape (n_valid, D), or None if all corrupt
    """
    all_vecs = []
    skipped  = []
    for start in range(0, len(paths), BATCH):
        batch_paths = paths[start:start + BATCH]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                tensors.append(preprocess(img))
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"[{context}] Image not found: {p}"
                )
            except Exception as e:
                print(f"CORRUPT IMAGE [{context}]: {os.path.basename(p)} "
                      f"— {type(e).__name__}: {e}")
                skipped.append(p)
                continue
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            vecs = model.encode_image(batch)
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        all_vecs.append(vecs.cpu().numpy())

    if skipped:
        print(f"SKIPPED {len(skipped)} corrupt image(s) in [{context}]")
    if not all_vecs:
        return None
    return np.vstack(all_vecs)


# properties that still need at least one modality
needs = [i for i in range(N)
         if not (has_text[i] and has_photos[i] and has_gsv[i] and has_sat[i])]
print(f"\n{len(needs)} properties need processing ({N - len(needs)} already done)")


# 1. Text
text_todo    = []
text_indices = []
for i in needs:
    if has_text[i]:
        continue
    desc = df.iloc[i].get('listing_description', None)
    if isinstance(desc, str) and len(desc.strip()) > 10:
        text_todo.append(desc)
        text_indices.append(i)

if text_todo:
    print(f"\nEncoding {len(text_todo)} text descriptions:")
    for start in range(0, len(text_todo), 256):
        batch_texts = text_todo[start:start + 256]
        batch_idx = text_indices[start:start + 256]
        tokens = clip.tokenize(batch_texts, truncate=True).to(device)
        with torch.no_grad():
            vecs = model.encode_text(tokens)
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        vecs_np = vecs.cpu().numpy()
        for j, idx in enumerate(batch_idx):
            emb_text[idx] = vecs_np[j]
            has_text[idx] = True
        print(f"Text: {min(start + 256, len(text_todo))}/{len(text_todo)}")
    checkpoint()

print(f"Text done: {has_text.sum()}/{N}")


# 2. Listing photos
photo_todo  = sum(1 for i in needs if not has_photos[i])
photo_count = 0
progress = make_progress_tracker("Photos", photo_todo)
print(f"\nEncoding listing photos ({photo_todo} remaining):")

for i in needs:
    if has_photos[i]:
        continue
    prop_id = str(df.iloc[i]['id'])
    pdir = f'{PHOTOS_DIR}/{prop_id}'

    if not os.path.isdir(pdir):
        print(f"WARNING [{prop_id}]: Listing photos folder not found, skipping.")
        continue

    photo_files = sorted([
        f'{pdir}/{f}' for f in os.listdir(pdir)
        if f.lower().endswith('.jpg')
    ])
    if not photo_files:
        print(f"WARNING [{prop_id}]: No .jpg files in lisitng photos folder, skipping.")
        continue

    t0 = time.time()
    vecs = encode_images_batched(photo_files, context=f"Photos {prop_id}")
    t_encode = time.time() - t0

    print(f"[{prop_id}] {len(photo_files)} photos | encode={t_encode:.2f}s")

    if vecs is None:
        print(f"WARNING [{prop_id}]: All photo files were corrupt, skipping property.")
        continue

    avg = vecs.mean(axis=0)
    emb_photos[i] = avg / (np.linalg.norm(avg) + 1e-8)
    has_photos[i] = True
    photo_count += 1
    progress(photo_count)

    if photo_count % 100 == 0:
        checkpoint()

print(f"Photos done: {has_photos.sum()}/{N}")
checkpoint()


# street view
gsv_todo  = sum(1 for i in needs if not has_gsv[i])
gsv_count = 0
progress = make_progress_tracker("GSV", gsv_todo)
print(f"\nEncoding street view images ({gsv_todo} remaining):")

for i in needs:
    if has_gsv[i]:
        continue
    prop_id = str(df.iloc[i]['id'])
    row = df.iloc[i]
    local_paths = []

    for h in [0, 90, 180, 270]:
        col = f'gsv_file_{h}'
        drive_path = row.get(col, None)
        if not isinstance(drive_path, str):
            print(f"WARNING [{prop_id}]: Missing path for GSV angle {h}, skipping angle.")
            continue
        fname = os.path.basename(drive_path)
        local_path = f'{GSV_DIR}/{fname}'
        if not os.path.exists(local_path):
            print(f"WARNING [{prop_id}]: GSV file not found: {fname}, skipping angle.")
            continue
        local_paths.append(local_path)

    if not local_paths:
        print(f"WARNING [{prop_id}]: No valid GSV images found, skipping property.")
        continue

    t0 = time.time()
    vecs = encode_images_batched(local_paths, context=f"GSV {prop_id}")
    t_encode = time.time() - t0

    print(f"[{prop_id}] {len(local_paths)} angles | encode={t_encode:.2f}s")

    if len(vecs) != len(local_paths):
        raise RuntimeError(
            f"[{prop_id}] GSV vector count mismatch: "
            f"expected {len(local_paths)}, got {len(vecs)}. "
            f"Paths: {local_paths}"
        )

    avg = np.mean(vecs, axis=0)
    emb_gsv[i] = avg / (np.linalg.norm(avg) + 1e-8)
    has_gsv[i] = True
    gsv_count += 1
    progress(gsv_count)

    if gsv_count % 100 == 0:
        checkpoint()

print(f"GSV done: {has_gsv.sum()}/{N}")
checkpoint()


# satellite
sat_todo = sum(1 for i in needs if not has_sat[i])
sat_count = 0
progress = make_progress_tracker("Satellite", sat_todo)
print(f"\nEncoding satellite images ({sat_todo} remaining):")

for i in needs:
    if has_sat[i]:
        continue
    prop_id = str(df.iloc[i]['id'])
    drive_path = df.iloc[i].get('sat_file', None)

    if not isinstance(drive_path, str):
        print(f"WARNING [{prop_id}]: Missing path for satellite image, skipping.")
        continue

    fname = os.path.basename(drive_path)
    local_path = f'{SAT_DIR}/{fname}'

    if not os.path.exists(local_path):
        print(f"WARNING [{prop_id}]: Satellite file not found: {fname}, skipping.")
        continue

    t0 = time.time()
    vecs = encode_images_batched([local_path], context=f"Satellite {prop_id}")
    t_encode = time.time() - t0

    print(f"[{prop_id}] encode={t_encode:.2f}s")

    if vecs is None:
        print(f"WARNING [{prop_id}]: Satellite image was corrupt, skipping.")
        continue

    if len(vecs) != 1:
        raise RuntimeError(
            f"[{prop_id}] Satellite expected 1 vector, got {len(vecs)}."
        )

    emb_sat[i] = vecs[0]
    has_sat[i] = True
    sat_count += 1
    progress(sat_count)

    if sat_count % 100 == 0:
        checkpoint()

print(f"Satellite done: {has_sat.sum()}/{N}")

# final save
checkpoint()
print(f"\ndone! Saved to {out_path}")
print(f"Text: {has_text.sum()}/{N}")
print(f"Photos: {has_photos.sum()}/{N}")
print(f"GSV: {has_gsv.sum()}/{N}")
print(f"Satellite: {has_sat.sum()}/{N}")
