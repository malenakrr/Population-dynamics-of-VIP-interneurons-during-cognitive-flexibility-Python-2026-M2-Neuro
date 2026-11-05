# The following code is a Python script that implements a full processing pipeline for 1-photon calcium imaging data.
# It uses the CaImAn library to perform motion correction, source extraction and quality evaluation.
# It is based on a pre-existing pipeline, but has been modified to include additional features (produce specific outputs).
# It also optimizes the parameters for 1-photon miniscope data.
# A. Giovannucci et al., (2019), CaImAn an open source tool for scalable calcium imaging data analysis, eLife

# Imports
import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import csv
import glob
import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tifffile
from scipy import sparse
from scipy.ndimage import (binary_dilation, binary_fill_holes, label, median_filter,
                           percentile_filter, uniform_filter1d)

import caiman
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.summary_images import correlation_pnr

# Miniscope video pathway
VIDEO_PATH = ("/Users/noodle/Desktop/Master Neuro UPC /Master 2/Python/Rule reversal/"
              "10_15_52/miniscopeDeviceName/0.avi")

DECAY_TIME = 1.3          # s, GCaMP decay time (tau)
FR_FALLBACK = 60.0        # Hz, used only if the frame rate cannot be read from the recording folder

# Motion correction
GSIG_FILT = 3             # px, spatial high-pass for 1p registration (~ cell radius)
MAX_SHIFT = 15            # px, max rigid shift allowed 
STRIDES = (96, 96)        # px, pw-rigid patch grid
OVERLAPS = (48, 48)       # px, overlap between pw-rigid patches
MAX_DEVIATION_RIGID = 5   # px, max deviation of a patch from the rigid shift
NITER_RIG = 2             # template refinement iterations in the rigid pass

# Motion frame flagging
MOTION_DEV_THR_PX = 2.5   # px, a frame is flagged if its shift magnitude exceeds the running median by more than this
MOTION_PAD_S = 0.1        # s, flagged epochs are extended by this much on both sides

# CNMF-E (from CaImAn demo)
GSIG = 3                  # px, half-width of a neuron 
GSIZ = 4 * GSIG + 1       # px, neuron bounding-box size (odd)
MIN_CORR = 0.8            # initialisation threshold on the correlation image
MIN_PNR = 10              # initialisation threshold on the peak-to-noise-ratio image
MIN_SNR = 2.5             # component evaluation
RVAL_THR = 0.85           # component evaluation (spatial correlation)
CORR_PNR_STEP = 2         # use every n-th frame for correlation/PNR images (saves memory)

# ROI masks and curation
MASK_THRESHOLD = 0.2      # ROI mask = pixels > this fraction of the footprint maximum
MIN_AREA_FACTOR = 0.25    # remove ROIs smaller than this x the median ROI area
MAX_AREA_FACTOR = 2.5     # remove ROIs larger than this x the median ROI area (merged cells, vessels)
MIN_SOLIDITY = 0.80       # remove irregular ROIs : mask area / convex-hull area below this
BORDER_MARGIN_PX = 3      # remove ROIs that come closer than this to the image edge
MIN_REL_INTENSITY = 0.30  # remove ROIs whose mean raw fluorescence is below this x the median over ROIs 
MIN_TRACE_SNR = 3.0       # remove ROIs whose dF/F (99th pct - median) / noise is below this 
MERGE_MIN_OVERLAP = 0.25  # merge two ROIs if overlap (pixels in common / smaller area) is at least this ...
MERGE_MIN_CORR = 0.80     # ... AND their (0.25 s smoothed) dF/F traces correlate at least this much

# Trace extraction (dF/F)
NEUROPIL_GAP = 2          # px, gap between ROI and annulus
NEUROPIL_WIDTH = 10       # px, annulus width
MIN_NEUROPIL_PIX = 50     # below this the annulus is replaced by all non-ROI pixels
NEUROPIL_R = 0.7          # fixed neuropil factor for the 'annulus' dF/F variant
BASELINE_PERCENTILE = 8   # F0 = running percentile ...
BASELINE_WINDOW_S = 30    # ... over this window (s)
DFF_METHOD = "demixed"    


# Functions

def parse_args():
    p = argparse.ArgumentParser(description="CNMF-E pipeline for NDNF miniscope data")
    p.add_argument("--video", default=VIDEO_PATH, help="Miniscope .avi file")
    p.add_argument("--out", default=None, help="Output folder (default: <video folder>/caiman_output)")
    p.add_argument("--fr", type=float, default=None, help="Frame rate in Hz (default: read from recording folder)")
    p.add_argument("--nproc", type=int, default=None, help="Number of worker processes (default: automatic)")
    p.add_argument("--dff_method", default=DFF_METHOD, choices=["demixed", "annulus", "regression"],
                   help="dF/F variant used for dFF_traces.csv and the figures (all three are saved in the npz)")
    return p.parse_args()


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def mean_std_frames(movie, chunk=100):
    s = np.zeros(movie.shape[1:], np.float64)
    s2 = np.zeros_like(s)
    for i in range(0, len(movie), chunk):
        x = movie[i:i + chunk].astype(np.float64)
        s += x.sum(0)
        s2 += (x ** 2).sum(0)
    n = len(movie)
    mean = s / n
    return mean, np.sqrt(np.maximum(s2 / n - mean ** 2, 0))


def mean_std_rows(Yr, chunk=20000):
    n = Yr.shape[0]
    mean = np.empty(n, np.float32)
    std = np.empty(n, np.float32)
    for p0 in range(0, n, chunk):
        x = np.asarray(Yr[p0:p0 + chunk], dtype=np.float32)
        mean[p0:p0 + chunk] = x.mean(1)
        std[p0:p0 + chunk] = x.std(1)
    return mean, std


def estimate_frame_rate(video_path, fps_meta):
    folder = Path(video_path).parent
    try:
        ts = np.genfromtxt(folder / "timeStamps.csv", delimiter=",", skip_header=1)
        dt = np.median(np.diff(ts[:, 1]))          # column 2 = time stamp (ms)
        if dt > 0:
            return 1000.0 / dt, "timeStamps.csv (median frame interval)"
    except Exception:
        pass
    try:
        meta = json.load(open(folder / "metaData.json"))
        fr = float(re.findall(r"[\d.]+", str(meta["frameRate"]))[0])
        if fr > 0:
            return fr, "metaData.json"
    except Exception:
        pass
    if fps_meta and 1 < fps_meta < 200:
        return float(fps_meta), "AVI header"
    return FR_FALLBACK, "fallback value"


def avi_to_tiff(video_path, tif_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open {video_path}")
    fps_meta = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:          
            frame = frame[..., 0]
        frames.append(frame)
    cap.release()
    if not frames:
        raise IOError(f"No frames could be read from {video_path}")
    movie = np.stack(frames)
    del frames
    raw_mean, raw_std = mean_std_frames(movie)
    tifffile.imwrite(tif_path, movie)
    return movie.shape, fps_meta, raw_mean, raw_std


def build_params(tif_path, fr):
    return params.CNMFParams(params_dict={
        # data
        "fnames": [tif_path],
        "fr": fr,
        "decay_time": DECAY_TIME,
        # motion correction
        "pw_rigid": True,
        "gSig_filt": (GSIG_FILT, GSIG_FILT),
        "max_shifts": (MAX_SHIFT, MAX_SHIFT),
        "strides": STRIDES,
        "overlaps": OVERLAPS,
        "max_deviation_rigid": MAX_DEVIATION_RIGID,
        "niter_rig": NITER_RIG,
        "num_frames_split": 100,
        "upsample_factor_grid": 4,
        "border_nan": "copy",
        "nonneg_movie": True,
        "shifts_opencv": True,
        # CNMF-E
        "method_init": "corr_pnr",
        "K": None,
        "gSig": (GSIG, GSIG),
        "gSiz": (GSIZ, GSIZ),
        "merge_thr": 0.7,
        "p": 1,
        "tsub": 2,
        "ssub": 1,
        "rf": 40,
        "stride": 20,
        "only_init": True,
        "nb": 0,
        "nb_patch": 0,
        "low_rank_background": None,
        "update_background_components": True,
        "min_corr": MIN_CORR,
        "min_pnr": MIN_PNR,
        "normalize_init": False,
        "center_psf": True,
        "ssub_B": 2,
        "ring_size_factor": 1.4,
        "del_duration": 1,
        # Component evaluation
        "min_SNR": MIN_SNR,
        "rval_thr": RVAL_THR,
        "use_cnn": False,     
    })


def run_motion_correction(opts, dview, fname):
    motion = opts.get_group("motion")
    mc_rig = MotionCorrect([fname], dview=dview, **dict(motion, pw_rigid=False)) # First, do rigid
    mc_rig.motion_correct(save_movie=False)
    mc_pw = MotionCorrect([fname], dview=dview, **dict(motion, pw_rigid=True)) # Then do piecewise rigid
    mc_pw.motion_correct(save_movie=True, template=mc_rig.total_template_rig)
    return mc_rig, mc_pw


# Motion frame flagging : flag frames with fast motion bursts
# A frame is flagged when magnitude exceeds its own running median by more than MOTION_DEV_THR_PX
def flag_motion_frames(mc_pw, fr, T):
    xs = np.asarray(mc_pw.x_shifts_els, dtype=float)
    ys = np.asarray(mc_pw.y_shifts_els, dtype=float)
    xs, ys = xs.reshape(len(xs), -1), ys.reshape(len(ys), -1)
    mag = np.sqrt(xs ** 2 + ys ** 2).max(axis=1)
    win = int(2 * fr) | 1                               
    dev = mag - median_filter(mag, size=win, mode="nearest")
    flag = dev > MOTION_DEV_THR_PX
    if MOTION_ABS_THR_PX is not None:
        flag |= mag > MOTION_ABS_THR_PX
    pad = int(round(MOTION_PAD_S * fr))
    if pad > 0:
        flag = binary_dilation(flag, structure=np.ones(2 * pad + 1, bool))
    if len(flag) != T:
        print(f"  WARNING: {len(flag)} shift entries for {T} frames, motion flagging skipped")
        flag, mag, dev = np.zeros(T, bool), np.zeros(T), np.zeros(T)
    return flag, mag, dev


def interpolate_bad(X, bad):
    if not bad.any():
        return X.copy()
    good = np.flatnonzero(~bad)
    idx = np.arange(X.shape[1])
    return np.stack([np.interp(idx, good, x[good]) for x in X])


def nan_bad(X, bad):
    return np.where(bad[None, :], np.nan, X)


def shade_runs(ax, t, bad):
    if bad is None or not np.any(bad):
        return
    d = np.diff(np.concatenate([[0], np.asarray(bad, int), [0]]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        ax.axvspan(t[s], t[e - 1], color="red", alpha=0.12, lw=0)


# ROI masks and curation
def build_masks(A, idx, dims, thr):
    masks = np.zeros((len(idx),) + tuple(dims), dtype=bool)
    for j, k in enumerate(idx):
        col = A[:, k].toarray().ravel()
        m = (col > thr * col.max()).reshape(dims, order="F")
        lab, n = label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            m = lab == sizes.argmax()
        masks[j] = binary_fill_holes(m)
    return masks


def solidity(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.int32)
    hull = cv2.convexHull(pts)
    hm = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(hm, hull, 1)
    return float(m.sum()) / max(int(hm.sum()), 1)


def trace_snr(X):
    #(99th percentile - median) / robust noise, noise from the frame-to-frame differences
    sig = np.percentile(X, 99, axis=1) - np.median(X, axis=1)
    d = np.diff(X, axis=1)
    noise = 1.4826 * np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1) / np.sqrt(2)
    return sig / np.maximum(noise, 1e-9)


def curate_rois(masks, F_raw, dff, fr):
    #Remove bad ROIs, then merge duplicates
    K, (H, W) = len(masks), masks.shape[1:]
    area = masks.reshape(K, -1).sum(1).astype(float)
    sol = np.array([solidity(m) for m in masks])
    border = np.empty(K)
    for k, m in enumerate(masks):
        rr, cc = np.nonzero(m)
        border[k] = min(rr.min(), cc.min(), H - 1 - rr.max(), W - 1 - cc.max())
    inten = F_raw.mean(axis=1)
    snr = trace_snr(dff)
    med_area, med_int = np.median(area), np.median(inten)

    reason = np.array([""] * K, dtype=object)
    for k in range(K):
        if area[k] < MIN_AREA_FACTOR * med_area:
            reason[k] = "too small"
        elif area[k] > MAX_AREA_FACTOR * med_area:
            reason[k] = "too large"
        elif sol[k] < MIN_SOLIDITY:
            reason[k] = "irregular shape"
        elif border[k] < BORDER_MARGIN_PX:
            reason[k] = "at image border"
        elif inten[k] < MIN_REL_INTENSITY * med_int:
            reason[k] = "dim region"
        elif snr[k] < MIN_TRACE_SNR:
            reason[k] = "low SNR"
    keep = np.flatnonzero(reason == "")
    metrics = dict(area=area, solidity=sol, border_dist=border, mean_raw=inten, snr=snr)
    if len(keep) == 0:
        return [], reason, metrics

    # Merge duplicates : overlapping masks AND correlated traces
    n = len(keep)
    Mf = sparse.csr_matrix(masks[keep].reshape(n, -1)).astype(np.int32)
    inter = (Mf @ Mf.T).toarray().astype(float)
    a = np.diag(inter)
    overlap = inter / np.maximum(np.minimum.outer(a, a), 1)
    sm = uniform_filter1d(dff[keep], size=max(int(0.25 * fr), 1), axis=1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(sm)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = np.argwhere(np.triu((overlap >= MERGE_MIN_OVERLAP) & (corr >= MERGE_MIN_CORR), 1))
    for i, j in pairs:
        parent[find(j)] = find(i)
    groups_d = {}
    for i in range(n):
        groups_d.setdefault(find(i), []).append(int(keep[i]))
    groups = sorted(groups_d.values(), key=lambda g: g[0])
    return groups, reason, metrics


# Trace extraction : mean over ROI pixels and neuropil annulus
def neuropil_pixels(masks, dims, bord_px, extra_exclude=None):
    union = masks.any(axis=0)
    if extra_exclude is not None:
        union = union | extra_exclude.any(axis=0)
    excl = binary_dilation(union, structure=disk(NEUROPIL_GAP)) if NEUROPIL_GAP > 0 else union.copy()
    if bord_px > 0:                                  
        border = np.ones(dims, bool)
        border[bord_px:-bord_px, bord_px:-bord_px] = False
        excl |= border
    global_pix = np.flatnonzero((~excl).reshape(-1, order="F"))
    se = disk(NEUROPIL_GAP + NEUROPIL_WIDTH)
    pad = NEUROPIL_GAP + NEUROPIL_WIDTH + 1
    out, n_fallback = [], 0
    for m in masks:
        rr, cc = np.nonzero(m)
        r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad + 1, dims[0])
        c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad + 1, dims[1])
        ring = binary_dilation(m[r0:r1, c0:c1], structure=se) & ~excl[r0:r1, c0:c1]
        ri, ci = np.nonzero(ring)
        pix = np.ravel_multi_index((ri + r0, ci + c0), dims, order="F")
        if pix.size < MIN_NEUROPIL_PIX:
            pix, n_fallback = global_pix, n_fallback + 1
        out.append(pix)
    if n_fallback:
        print(f"  {n_fallback} ROI(s) had too small an annulus: all non-ROI pixels were used instead")
    return out


def mean_traces(Yr, pixel_lists, chunk=20000):
    # Mean fluorescence over each pixel list
    K, (npix, T) = len(pixel_lists), Yr.shape
    rows = np.concatenate([np.full(len(p), k) for k, p in enumerate(pixel_lists)])
    cols = np.concatenate(pixel_lists)
    vals = np.concatenate([np.full(len(p), 1.0 / len(p)) for p in pixel_lists])
    M = sparse.csc_matrix((vals, (rows, cols)), shape=(K, npix))
    out = np.zeros((K, T), np.float64)
    for p0 in range(0, npix, chunk):
        sub = M[:, p0:p0 + chunk]
        if sub.nnz:
            out += sub @ np.asarray(Yr[p0:p0 + chunk], dtype=np.float32)
    return out


def extract_traces(Yr, masks, dims, bord_px, extra_exclude=None):
    roi_pix = [np.flatnonzero(m.reshape(-1, order="F")) for m in masks]
    neu_pix = neuropil_pixels(masks, dims, bord_px, extra_exclude)
    both = mean_traces(Yr, roi_pix + neu_pix)
    return both[:len(masks)], both[len(masks):], neu_pix


# dF/F : different methods were tested
def running_baseline(X, fr):
    # Running-percentile baseline F0(t) of each row of X.
    # Computed on a ~5 Hz block-averaged copy then interpolated back to every frame
    K, T = X.shape
    ds = max(1, int(round(fr / 5.0)))
    nb = T // ds
    if nb < 3:
        return np.repeat(np.percentile(X, BASELINE_PERCENTILE, axis=1, keepdims=True), T, axis=1)
    Xd = X[:, :nb * ds].reshape(K, nb, ds).mean(axis=2)
    win = int(min(max(BASELINE_WINDOW_S * fr / ds, 3), nb))
    B = percentile_filter(Xd, BASELINE_PERCENTILE, size=(1, win), mode="nearest")
    B = uniform_filter1d(B, size=max(win // 3, 1), axis=1, mode="nearest")
    tc = (np.arange(nb) + 0.5) * ds - 0.5
    t = np.arange(T)
    return np.stack([np.interp(t, tc, b) for b in B])


def dff_annulus(F_raw, F_neu, fr, r):
    # F_corr = F_raw - r * (F_neu - median(F_neu))   (keeps the baseline level, removes neuropil fluctuations).
    F_corr = F_raw - r * (F_neu - np.median(F_neu, axis=1, keepdims=True))
    F0 = np.maximum(running_baseline(F_corr, fr), 1e-6)
    return F_corr, F0, (F_corr - F0) / F0


def dff_demixed(F_dem, F_raw, fr):
    # (demixed trace - its own baseline) / baseline of the raw ROI fluorescence.
    # The demixed trace is background-free, the denominator keeps the ROI's real brightness, so this is
    # conservative (background in the denominator) but free of background fluctuations in the numerator
    delta = F_dem - running_baseline(F_dem, fr)
    F0 = np.maximum(running_baseline(F_raw, fr), 1e-6)
    return delta / F0


def estimate_neuropil_r(F_raw, F_neu, F_dem, fr):
    # Per-ROI neuropil factor r_k in [0, 1].
    # What is left in the ROI once CNMF-E's demixed cell signal is removed (F_raw - F_dem) is background /
    # neuropil plus noise, so its regression on the annulus trace gives the contamination factor without
    # being corrupted by the cell's own transients. 
    win = max(int(5 * fr), 3)
    y = F_raw - F_dem
    y = y - uniform_filter1d(y, size=win, axis=1, mode="nearest")
    x = F_neu - uniform_filter1d(F_neu, size=win, axis=1, mode="nearest")
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    r = (xc * yc).sum(axis=1) / np.maximum((xc * xc).sum(axis=1), 1e-12)
    return np.clip(r, 0.0, 1.0)


def demixed_traces(A, C, YrA, idx, groups, final_masks, dims):
    # CNMF-E demixed trace (C + YrA) of each final ROI, converted to mean-fluorescence units:
    # contribution of component k to the mean over the ROI pixels = C_k * mean(A_k over those pixels)
    out = np.zeros((len(groups), C.shape[1]))
    for g, (members, m) in enumerate(zip(groups, final_masks)):
        pix = np.flatnonzero(m.reshape(-1, order="F"))
        for k in members:
            comp = idx[k]
            w = A[:, comp].toarray().ravel()[pix].mean()
            out[g] += w * (C[comp] + (YrA[comp] if YrA is not None else 0.0))
    return out


def process_traces(est, idx, Yr, dims, bord_px, fr, bad, method):
    # Masks -> curation -> final traces -> dF/F variants
    A = est.A.tocsc()
    C = np.asarray(est.C)
    YrA = None if est.YrA is None else np.asarray(est.YrA)

    # 1) candidate masks and a first pass of traces, only used to curate
    masks_all = build_masks(A, idx, dims, MASK_THRESHOLD)
    F_raw, F_neu, _ = extract_traces(Yr, masks_all, dims, bord_px)
    _, _, dff_pre = dff_annulus(interpolate_bad(F_raw, bad), interpolate_bad(F_neu, bad), fr, NEUROPIL_R)
    groups, reason, metrics = curate_rois(masks_all, F_raw, dff_pre, fr)
    n_removed = int((reason != "").sum())
    n_merged = sum(len(g) - 1 for g in groups)
    print(f"  curation: {len(masks_all)} candidates -> {n_removed} removed "
          f"({', '.join(f'{r}: {int((reason == r).sum())}' for r in sorted(set(reason) - {''}))}), "
          f"{n_merged} merged into other ROIs -> {len(groups)} final ROIs")
    if not groups:
        raise RuntimeError("Curation removed every ROI: relax the MIN_/MAX_ curation settings.")
    final_masks = np.stack([masks_all[g].any(axis=0) for g in groups])

    # 2) final traces (annulus excludes every candidate ROI, also the removed ones)
    F_raw, F_neu, neu_pix = extract_traces(Yr, final_masks, dims, bord_px, extra_exclude=masks_all)
    F_raw_i, F_neu_i = interpolate_bad(F_raw, bad), interpolate_bad(F_neu, bad)
    F_dem = demixed_traces(A, C, YrA, idx, groups, final_masks, dims)
    F_dem_i = interpolate_bad(F_dem, bad)
    r_reg = estimate_neuropil_r(F_raw_i, F_neu_i, F_dem_i, fr)

    F_corr, F0, dff_a = dff_annulus(F_raw_i, F_neu_i, fr, NEUROPIL_R)
    _, _, dff_r = dff_annulus(F_raw_i, F_neu_i, fr, r_reg[:, None])
    dff_d = dff_demixed(F_dem_i, F_raw_i, fr)
    variants = {"demixed": nan_bad(dff_d, bad), "annulus": nan_bad(dff_a, bad), "regression": nan_bad(dff_r, bad)}
    return dict(masks_all=masks_all, final_masks=final_masks, groups=groups, reason=reason, metrics=metrics,
                F_raw=F_raw, F_neu=F_neu, neu_pix=neu_pix, F_dem=F_dem, F_corr=F_corr, F0=F0, r_reg=r_reg,
                variants=variants, dff=variants[method])


# Figures 
def stretch(img, lo=1, hi=99.5):
    return np.percentile(img, [lo, hi])


def save_before_after(raw_mean, raw_std, reg_mean, reg_std, out_dir):
    panels = [("Raw mean (before)", raw_mean, "before_mean"),
              ("Registered mean (after)", reg_mean, "after_mean"),
              ("Raw temporal SD (before)", raw_std, "before_std"),
              ("Registered temporal SD (after)", reg_std, "after_std")]
    H, W = raw_mean.shape
    fig, axs = plt.subplots(2, 2, figsize=(13, 13 * H / W))
    for ax, (title, img, name) in zip(axs.ravel(), panels):
        v0, v1 = stretch(img)
        ax.imshow(img, cmap="gray", vmin=v0, vmax=v1)
        ax.set_title(title)
        ax.axis("off")
        plt.imsave(out_dir / f"{name}.png", img, cmap="gray", vmin=v0, vmax=v1)   # native resolution
    fig.tight_layout()
    fig.savefig(out_dir / "window_before_after.png", dpi=200)
    plt.close(fig)


def save_corr_pnr(cn_filter, pnr, out_dir):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    for ax, img, title in zip(axs, (cn_filter, pnr), ("Correlation image", "Peak-to-noise ratio")):
        im = ax.imshow(img, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "corr_pnr.png", dpi=150)
    plt.close(fig)


def save_overlay(img, masks, out_path, title):
    # Filled semi-transparent masks + contours + ROI numbers on top of an image
    H, W = img.shape
    fig, ax = plt.subplots(figsize=(10, 10 * H / W))
    v0, v1 = stretch(img)
    ax.imshow(img, cmap="gray", vmin=v0, vmax=v1)
    rng = np.random.default_rng(0)
    colors = plt.cm.hsv(rng.permutation(len(masks)) / max(len(masks), 1))
    overlay = np.zeros((H, W, 4))
    for m, col in zip(masks, colors):
        overlay[m] = (*col[:3], 0.40)
    ax.imshow(overlay)
    for j, (m, col) in enumerate(zip(masks, colors)):
        ax.contour(m.astype(float), levels=[0.5], colors=[col[:3]], linewidths=0.8)
        rr, cc = np.nonzero(m)
        ax.text(cc.mean(), rr.mean(), str(j), color="white", fontsize=7, ha="center", va="center")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_curation_overlay(img, masks_all, reason, groups, out_path):
    # Kept ROIs in green, removed ROIs coloured by reason
    H, W = img.shape
    palette = {"too small": "tab:red", "too large": "tab:orange", "irregular shape": "magenta",
               "at image border": "yellow", "dim region": "deepskyblue", "low SNR": "white"}
    kept = {k for g in groups for k in g}
    fig, ax = plt.subplots(figsize=(10, 10 * H / W))
    v0, v1 = stretch(img)
    ax.imshow(img, cmap="gray", vmin=v0, vmax=v1)
    for k, m in enumerate(masks_all):
        col = "lime" if k in kept else palette.get(reason[k], "red")
        ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=0.9)
    handles = [plt.Line2D([], [], color="lime", label=f"kept ({len(kept)} candidates -> {len(groups)} ROIs)")]
    for r, col in palette.items():
        n = int((reason == r).sum())
        if n:
            handles.append(plt.Line2D([], [], color=col, label=f"{r} ({n})"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.8)
    ax.set_title("ROI curation")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_registration_qc(mc_rig, mc_pw, fr, bad, mag, dev, out_dir):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    shifts = np.asarray(mc_rig.shifts_rig, dtype=float)
    t = np.arange(len(shifts)) / fr
    axs[0].plot(t, shifts[:, 0], label="rows")
    axs[0].plot(t, shifts[:, 1], label="cols")
    axs[0].set_ylabel("rigid shift (px)")
    axs[0].legend()
    try:
        xs = np.abs(np.asarray(mc_pw.x_shifts_els, dtype=float))
        ys = np.abs(np.asarray(mc_pw.y_shifts_els, dtype=float))
        xs, ys = xs.reshape(len(xs), -1), ys.reshape(len(ys), -1)
        axs[1].plot(t, xs.max(1), label="max |x shift| over patches")
        axs[1].plot(t, ys.max(1), label="max |y shift| over patches")
        axs[1].legend()
    except Exception:
        pass
    axs[1].set_ylabel("pw-rigid shift (px)")
    axs[2].plot(t, mag, color="k", lw=0.8, label="shift magnitude (max over patches)")
    axs[2].plot(t, mag - dev, color="tab:blue", lw=0.8, label="running median (2 s)")
    axs[2].set_ylabel("magnitude (px)")
    axs[2].set_xlabel("time (s)")
    axs[2].legend(loc="upper left")
    for ax in axs:
        shade_runs(ax, t, bad)
    axs[2].set_title(f"{100 * bad.mean():.1f}% of frames flagged as motion (red), "
                     f"threshold = median + {MOTION_DEV_THR_PX} px", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "registration_qc.png", dpi=150)
    plt.close(fig)
    peak = np.abs(shifts).max()
    print(f"  max rigid shift: {peak:.1f} px (limit {MAX_SHIFT}); "
          f"{int(bad.sum())} frames flagged as motion ({100 * bad.mean():.1f}%)")
    if peak >= MAX_SHIFT - 0.5:
        print("  WARNING: rigid shifts reach MAX_SHIFT, registration may be clipped -> increase MAX_SHIFT")
    if bad.mean() > 0.15:
        print("  WARNING: more than 15% of frames flagged, consider raising MOTION_DEV_THR_PX")


def save_trace_figures(t, F_raw, F_neu, dff, out_dir, bad=None, n_examples=5):
    K = len(F_raw)
    fig, ax = plt.subplots(figsize=(12, 0.15 * K + 2))
    im = ax.imshow(np.ma.masked_invalid(dff), aspect="auto", cmap="magma", vmin=0,
                   vmax=np.nanpercentile(dff, 99), extent=[t[0], t[-1], K, 0])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ROI")
    fig.colorbar(im, ax=ax, label="dF/F (flagged motion frames blank)")
    fig.tight_layout()
    fig.savefig(out_dir / "dFF_heatmap.png", dpi=150)
    plt.close(fig)

    ex = np.unique(np.linspace(0, K - 1, min(n_examples, K)).astype(int))
    fig, axs = plt.subplots(len(ex), 2, figsize=(14, 2.2 * len(ex)), sharex=True, squeeze=False)
    for r, k in enumerate(ex):
        axs[r, 0].plot(t, F_raw[k], "k", lw=0.8, label="raw ROI")
        axs[r, 0].plot(t, F_neu[k], color="tab:orange", lw=0.8, label="neuropil")
        axs[r, 0].set_ylabel(f"ROI {k}")
        axs[r, 1].plot(t, dff[k], "tab:green", lw=0.8)
        axs[r, 1].set_ylabel("dF/F")
        shade_runs(axs[r, 0], t, bad)
        shade_runs(axs[r, 1], t, bad)
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].set_title("Raw and neuropil fluorescence (a.u.); red = flagged motion")
    axs[0, 1].set_title("dF/F")
    axs[-1, 0].set_xlabel("time (s)")
    axs[-1, 1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "example_traces.png", dpi=150)
    plt.close(fig)


def save_dff_comparison(t, variants, out_dir, bad=None, n_examples=5):
    names = list(variants)
    K = len(variants[names[0]])
    ex = np.unique(np.linspace(0, K - 1, min(n_examples, K)).astype(int))
    fig, axs = plt.subplots(len(ex), 1, figsize=(12, 2.3 * len(ex)), sharex=True, squeeze=False)
    for r, k in enumerate(ex):
        for name in names:
            axs[r, 0].plot(t, variants[name][k], lw=0.8, label=name)
        axs[r, 0].set_ylabel(f"ROI {k}\ndF/F")
        shade_runs(axs[r, 0], t, bad)
    axs[0, 0].legend(ncol=3, loc="upper right")
    axs[0, 0].set_title("dF/F variants (red = flagged motion)")
    axs[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "dFF_methods_comparison.png", dpi=150)
    plt.close(fig)


def save_stacked_traces(t, traces, out_path, ylabel, center=False, normalize=False, title=None, bad=None):
    # One line per cell, stacked vertically (ROI 0 on top)
    # center=True : subtract each trace's median (use for raw fluorescence, whose offsets differ between cells)
    # normalize=True : scale each trace to its own 1st-99th percentile range (shows timing, hides amplitude)
    # Otherwise a common scale is used, so amplitudes are comparable and a scale bar is drawn
    # NaN (flagged motion frames) leave gaps
    X = np.asarray(traces, dtype=float).copy()
    K = len(X)
    if center:
        X -= np.nanmedian(X, axis=1, keepdims=True)
    p1, p99 = np.nanpercentile(X, 1, axis=1), np.nanpercentile(X, 99, axis=1)
    rng_k = np.maximum(p99 - p1, 1e-9)
    if normalize:
        X = X / rng_k[:, None]
        spacing = 0.7
    else:
        spacing = 0.7 * np.percentile(rng_k, 90)
    fig, ax = plt.subplots(figsize=(14, float(np.clip(0.11 * K, 8, 40))))
    shade_runs(ax, t, bad)
    for k in range(K):
        ax.plot(t, X[k] + (K - 1 - k) * spacing, color="0.15", lw=0.45)
    ticks = np.arange(0, K, 10)
    ax.set_yticks((K - 1 - ticks) * spacing)
    ax.set_yticklabels(ticks)
    ax.set_ylim(-spacing, K * spacing)
    ax.set_xlim(t[0], t[-1] * 1.02)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ROI")
    ax.set_title(title or f"{K} cells - {ylabel}" + (" (each trace normalised)" if normalize else ""))
    if not normalize:
        bar = float(f"{spacing:.1g}")          # round to one significant digit
        x_bar = t[-1] * 1.01
        ax.plot([x_bar, x_bar], [0, bar], color="tab:red", lw=2, clip_on=False)
        ax.text(x_bar, bar / 2, f" {bar:g}", color="tab:red", va="center", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)


def save_trace_csv(path, t, traces):
    header = "time_s," + ",".join(f"ROI_{k}" for k in range(len(traces)))
    np.savetxt(path, np.column_stack([t, traces.T]), delimiter=",", header=header, comments="")


# Main
def main():
    args = parse_args()
    video = Path(args.video).expanduser()
    if not video.is_file():
        raise FileNotFoundError(
            f"Video not found: {video!r}\n"
            "Check the path (the folder 'Master Neuro UPC ' has a trailing space in your original path).")
    out_dir = Path(args.out).expanduser() if args.out else video.parent / "caiman_output"
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()
    work_dir = work_dir.resolve()

    logger = logging.getLogger("caiman")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(filename)s:%(funcName)20s():%(lineno)s] %(message)s"))
    logger.addHandler(handler)

    os.chdir(work_dir)       # CaImAn writes its memmaps and log files in the working directory

    # AVI -> TIFF 
    print("[0/6] Reading the AVI ...")
    tif_path = str(work_dir / (video.stem + ".tif"))
    shape, fps_meta, raw_mean, raw_std = avi_to_tiff(str(video), tif_path)
    fr, fr_src = (args.fr, "command line") if args.fr else estimate_frame_rate(str(video), fps_meta)
    print(f"  {shape[0]} frames of {shape[1]}x{shape[2]} px, frame rate = {fr:.2f} Hz ({fr_src}), "
          f"decay time = {DECAY_TIME} s")
    print("  >>> check that the frame rate is right (expected ~60 Hz): it sets the temporal scale of the deconvolution <<<")
    opts = build_params(tif_path, fr)

    c, dview, n_processes = caiman.cluster.setup_cluster(backend="multiprocessing", n_processes=args.nproc)
    try:
        # 1. motion correction + motion-frame flagging
        print("[1/6] Motion correction (rigid pass, then piece-wise rigid pass) ...")
        mc_rig, mc_pw = run_motion_correction(opts, dview, tif_path)
        bord_px = int(np.ceil(np.maximum(np.max(np.abs(mc_pw.x_shifts_els)),
                                         np.max(np.abs(mc_pw.y_shifts_els)))))
        bord_px = 0 if opts.motion["border_nan"] == "copy" else bord_px
        bad, mag, dev = flag_motion_frames(mc_pw, fr, shape[0])
        save_registration_qc(mc_rig, mc_pw, fr, bad, mag, dev, out_dir)
        fname_new = caiman.save_memmap(mc_pw.fname_tot_els, base_name="memmap_", order="C",
                                       border_to_0=bord_px)

        Yr, dims, T = caiman.load_memmap(fname_new)
        images = Yr.T.reshape((T,) + dims, order="F")
        opts.change_params(params_dict={"dims": dims, "border_pix": bord_px})
        if len(bad) != T:
            bad = np.zeros(T, bool)

        # before / after images
        reg_mean, reg_std = mean_std_rows(Yr)
        reg_mean = reg_mean.reshape(dims, order="F")
        reg_std = reg_std.reshape(dims, order="F")
        save_before_after(raw_mean, raw_std, reg_mean, reg_std, out_dir)

        # 2. CNMF-E
        print("[2/6] Correlation / PNR images and CNMF-E ...")
        cn_filter, pnr = correlation_pnr(images[::CORR_PNR_STEP], gSig=opts.init["gSig"][0], swap_dim=False)
        save_corr_pnr(cn_filter, pnr, out_dir)
        print(f"  min_corr = {opts.init['min_corr']}, min_pnr = {opts.init['min_pnr']} "
              "(adjust with corr_pnr.png if too few / too many cells are found)")

        cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=None, params=opts)
        cnm.fit(images)

        # 3. component evaluation
        print("[3/6] Evaluating components ...")
        cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
        est = cnm.estimates
        idx = np.asarray(est.idx_components)
        print(f"  total components: {est.C.shape[0]}, accepted: {len(idx)}")
        if len(idx) == 0:
            raise RuntimeError("No component accepted: lower MIN_PNR / MIN_CORR / MIN_SNR and rerun.")

        # 4. ROI curation + traces 
        print("[4/6] ROI curation, raw / neuropil traces and dF/F variants ...")
        res = process_traces(est, idx, Yr, dims, bord_px, fr, bad, args.dff_method)
        groups, final_masks, F_raw, F_neu = res["groups"], res["final_masks"], res["F_raw"], res["F_neu"]
        dff, K, t = res["dff"], len(res["groups"]), np.arange(T) / fr
        S = est.S if est.S is not None else None

        # 5. save 
        print("[5/6] Saving traces and tables ...")
        save_trace_csv(out_dir / "raw_traces.csv", t, F_raw)                 # unmasked
        save_trace_csv(out_dir / "neuropil_traces.csv", t, F_neu)            # unmasked
        save_trace_csv(out_dir / "dFF_traces.csv", t, dff)                   # motion frames = NaN
        np.savetxt(out_dir / "motion_frames.csv", np.column_stack([np.arange(T), t, mag, dev, bad]),
                   delimiter=",", comments="", header="frame,time_s,motion_px,deviation_px,flagged",
                   fmt=["%d", "%.4f", "%.3f", "%.3f", "%d"])
        m = res["metrics"]
        with open(out_dir / "roi_curation.csv", "w", newline="") as fh:       # one row per candidate component
            w = csv.writer(fh)
            w.writerow(["component_index", "status", "reason", "final_roi", "area_px", "solidity",
                        "border_dist_px", "mean_raw_fluorescence", "trace_snr"])
            final_of = {k: g for g, members in enumerate(groups) for k in members}
            for k in range(len(idx)):
                status = "removed" if res["reason"][k] else ("merged" if len(groups[final_of[k]]) > 1 else "kept")
                w.writerow([int(idx[k]), status, res["reason"][k], final_of.get(k, -1), int(m["area"][k]),
                            f"{m['solidity'][k]:.3f}", int(m["border_dist"][k]), f"{m['mean_raw'][k]:.2f}",
                            f"{m['snr'][k]:.2f}"])
        centroids = np.array([[np.nonzero(mk)[0].mean(), np.nonzero(mk)[1].mean()] for mk in final_masks])
        with open(out_dir / "roi_info.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["roi", "member_components", "centroid_row", "centroid_col", "area_px", "neuropil_px",
                        "neuropil_r_regression"])
            for g, members in enumerate(groups):
                w.writerow([g, ";".join(str(int(idx[k])) for k in members), f"{centroids[g, 0]:.1f}",
                            f"{centroids[g, 1]:.1f}", int(final_masks[g].sum()), len(res["neu_pix"][g]),
                            f"{res['r_reg'][g]:.3f}"])
        comp = [idx[g[0]] for g in groups]
        np.savez_compressed(
            out_dir / "traces_and_rois.npz",
            time_s=t, fr=fr, decay_time=DECAY_TIME, dff_method=args.dff_method, motion_flag=bad,
            motion_px=mag, masks=final_masks, centroids=centroids,
            member_components=np.array([";".join(str(int(idx[k])) for k in g) for g in groups]),
            F_raw=F_raw, F_neuropil=F_neu, F_demixed=res["F_dem"], F_corrected=res["F_corr"], F0=res["F0"],
            neuropil_r_regression=res["r_reg"], dFF=dff,
            dFF_demixed=res["variants"]["demixed"], dFF_annulus=res["variants"]["annulus"],
            dFF_regression=res["variants"]["regression"],
            cnmfe_C_denoised=est.C[comp],
            cnmfe_S_deconvolved=S[comp] if S is not None else np.nan)
        try:
            cnm.dims = dims
            cnm.save(str(out_dir / "cnmfe_estimates.hdf5"))
        except Exception as e:
            print(f"  (could not save the CaImAn hdf5 file: {e})")

        # 6. figures
        print("[6/6] Figures ...")
        save_trace_figures(t, F_raw, F_neu, dff, out_dir, bad=bad)
        save_dff_comparison(t, res["variants"], out_dir, bad=bad)
        save_stacked_traces(t, dff, out_dir / "all_cells_dFF_stacked.png", "dF/F", bad=bad)
        save_stacked_traces(t, dff, out_dir / "all_cells_dFF_stacked_normalized.png", "dF/F",
                            normalize=True, bad=bad)
        save_stacked_traces(t, nan_bad(F_raw, bad), out_dir / "all_cells_raw_stacked.png",
                            "raw fluorescence (a.u., median-subtracted)", center=True, bad=bad)
        save_overlay(reg_mean, final_masks, out_dir / "ROI_overlay_on_mean.png",
                     f"{K} detected interneurons on the registered mean image")
        save_overlay(cn_filter, final_masks, out_dir / "ROI_overlay_on_correlation.png",
                     f"{K} detected interneurons on the correlation image")
        save_curation_overlay(reg_mean, res["masks_all"], res["reason"], groups, out_dir / "ROI_curation_overlay.png")
        print(f"\nDone. {K} final ROIs. Results in: {out_dir}")
    finally:
        caiman.stop_server(dview=dview)
        os.chdir(out_dir)
        for f in glob.glob(str(work_dir / "*_LOG_*")) + glob.glob("*_LOG_*"):
            try:
                os.remove(f)
            except OSError:
                pass


if __name__ == "__main__":    
    main()
