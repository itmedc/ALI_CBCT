#!/usr/bin/env python3
"""
Convert TotalSegmentator output to ALI_CBCT training format.

Extracts BONE-ONLY cephalometric landmarks from TotalSegmentator
segmentation masks (skull, mandible, vertebrae) and prepares
multi-scale NIfTI scans with .mrk.json annotations for ALI_CBCT.

Usage:
    python prepare_totalseg.py --data_dir /path/to/data --output_dir /path/to/output

Expected input — one of two layouts:

  Layout A (segmentation subfolder per scan):
    data/
    ├── patient001.nii.gz
    ├── patient001_seg/          ← or patient001/segmentations/
    │   ├── skull.nii.gz
    │   ├── mandible.nii.gz
    │   ├── vertebrae_C2.nii.gz
    │   └── ...
    ├── patient002.nii.gz
    └── patient002_seg/

  Layout B (separate scan and seg directories):
    python prepare_totalseg.py \
        --scans_dir /path/to/scans \
        --segs_dir  /path/to/segmentations \
        --output_dir /path/to/output

Output (ALI_CBCT format):
    output/
    ├── patient001/
    │   ├── patient001_scan_sp1.nii.gz
    │   ├── patient001_scan_sp0-3.nii.gz
    │   ├── patient001_lm_CB.mrk.json
    │   ├── patient001_lm_L.mrk.json
    │   └── patient001_lm_U.mrk.json
    └── conversion_report.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# =====================================================================
# Required TotalSegmentator mask filenames
# =====================================================================

REQUIRED_MASKS = {
    "skull": "skull.nii.gz",
    "mandible": "mandible.nii.gz",
}

OPTIONAL_MASKS = {
    "vertebrae_C2": "vertebrae_C2.nii.gz",
    "vertebrae_C3": "vertebrae_C3.nii.gz",
    "vertebrae_C4": "vertebrae_C4.nii.gz",
}

# ALI_CBCT landmark → group
LANDMARK_GROUPS = {
    # CB — Cranial Base
    "Ba": "CB", "S": "CB", "N": "CB",
    "RPo": "CB", "LPo": "CB",
    "RFZyg": "CB", "LFZyg": "CB",
    "C2": "CB", "C3": "CB", "C4": "CB",
    # U — Upper (bone only, no teeth)
    "RInfOr": "U", "LInfOr": "U",
    "ROr": "U", "LOr": "U",
    "RMZyg": "U", "LMZyg": "U",
    "ANS": "U", "PNS": "U", "A": "U",
    # L — Lower (bone only, no teeth)
    "RCo": "L", "LCo": "L",
    "RGo": "L", "LGo": "L",
    "Me": "L", "Gn": "L", "Pog": "L", "PogL": "L", "B": "L",
    "RSig": "L", "LSig": "L",
}


# =====================================================================
# Affine helpers
# =====================================================================

def build_affine(image: sitk.Image):
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    M = direction @ np.diag(spacing)
    return origin, M


def voxels_to_physical(voxels_zyx: np.ndarray, origin: np.ndarray, M: np.ndarray):
    """(N,3) [z,y,x] voxel coords → (N,3) LPS physical coords."""
    xyz = voxels_zyx[:, ::-1].astype(np.float64)
    return origin[None, :] + xyz @ M.T


def subsample(coords: np.ndarray, max_pts: int = 80_000, seed: int = 42):
    if len(coords) <= max_pts:
        return coords
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(coords), max_pts, replace=False)
    return coords[idx]


# =====================================================================
# Landmark extraction — mandible
# =====================================================================

def extract_mandible_landmarks(mask_path: str) -> dict[str, list[float]]:
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    coords = np.argwhere(arr > 0)

    if len(coords) < 100:
        return {}

    origin, M = build_affine(img)
    coords = subsample(coords)
    phys = voxels_to_physical(coords, origin, M)

    lm: dict[str, list[float]] = {}
    x_center = np.median(phys[:, 0])

    # ---- Midline landmarks -----------------------------------------
    mid_tol = 3.0  # mm
    mid_mask = np.abs(phys[:, 0] - x_center) < mid_tol
    midline = phys[mid_mask]

    if len(midline) < 20:
        return lm

    # Me — most inferior midline point
    me_idx = midline[:, 2].argmin()
    lm["Me"] = midline[me_idx].tolist()
    z_me = midline[me_idx, 2]

    # Pog — most anterior midline chin point (inferior 15 mm from Me)
    chin_mask = midline[:, 2] < z_me + 15
    chin = midline[chin_mask]
    if len(chin) > 0:
        pog_idx = chin[:, 1].argmin()
        lm["Pog"] = chin[pog_idx].tolist()
        lm["PogL"] = chin[pog_idx].tolist()

    # Gn — midpoint of Me and Pog
    if "Me" in lm and "Pog" in lm:
        lm["Gn"] = [(lm["Me"][i] + lm["Pog"][i]) / 2 for i in range(3)]

    # B — most posterior midline point 5–20 mm above Pog
    if "Pog" in lm:
        z_pog = lm["Pog"][2]
        b_mask = (midline[:, 2] > z_pog + 5) & (midline[:, 2] < z_pog + 20)
        b_pts = midline[b_mask]
        if len(b_pts) > 0:
            lm["B"] = b_pts[b_pts[:, 1].argmax()].tolist()

    # ---- Lateral landmarks (per side) ------------------------------
    for side_label, x_sign in [("R", -1), ("L", 1)]:
        if x_sign == -1:
            side_mask = phys[:, 0] < x_center - 20
        else:
            side_mask = phys[:, 0] > x_center + 20

        side = phys[side_mask]
        if len(side) < 50:
            continue

        # Go — most inferior-posterior point of mandibular angle
        go_score = side[:, 1] - side[:, 2]
        lm[f"{side_label}Go"] = side[go_score.argmax()].tolist()

        # Co — most superior point in posterior ramus
        post_thresh = np.percentile(side[:, 1], 75)
        ramus = side[side[:, 1] > post_thresh]
        if len(ramus) > 0:
            lm[f"{side_label}Co"] = ramus[ramus[:, 2].argmax()].tolist()

        # Sig — sigmoid notch: lowest z between Co and coronoid
        if f"{side_label}Co" in lm:
            z_co = lm[f"{side_label}Co"][2]
            upper_mask = side[:, 2] > z_co - 15
            upper = side[upper_mask]
            ant_post_mid = np.median(upper[:, 1])
            notch_mask = np.abs(upper[:, 1] - ant_post_mid) < 5
            notch = upper[notch_mask]
            if len(notch) > 5:
                lm[f"{side_label}Sig"] = notch[notch[:, 2].argmin()].tolist()

    return lm


# =====================================================================
# Landmark extraction — skull
# =====================================================================

def extract_skull_landmarks(mask_path: str) -> dict[str, list[float]]:
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    coords = np.argwhere(arr > 0)

    if len(coords) < 500:
        return {}

    origin, M = build_affine(img)
    coords = subsample(coords, 150_000)
    phys = voxels_to_physical(coords, origin, M)

    lm: dict[str, list[float]] = {}
    x_center = np.median(phys[:, 0])

    mid_tol = 3.0
    mid_mask = np.abs(phys[:, 0] - x_center) < mid_tol
    midline = phys[mid_mask]

    # -- Ba (Basion): most anterior-inferior edge of foramen magnum ---
    # The skull base has a gap (foramen magnum). Basion is at its anterior
    # midline edge. Strategy: find the most inferior midline skull points,
    # then among those find the most anterior one.
    if len(midline) > 50:
        z_5pct = np.percentile(midline[:, 2], 5)
        base_mask = midline[:, 2] < z_5pct + 10
        base_pts = midline[base_mask]
        if len(base_pts) > 10:
            # Foramen magnum region: posterior part of skull base
            post_thresh = np.percentile(base_pts[:, 1], 60)
            fm_pts = base_pts[base_pts[:, 1] > post_thresh]
            if len(fm_pts) > 5:
                lm["Ba"] = fm_pts[fm_pts[:, 1].argmin()].tolist()

    # -- N (Nasion): bridge of the nose, midline ---------------------
    # Most anterior midline point above the nasal aperture.
    # The nasal area is anterior + roughly middle height of the skull.
    if len(midline) > 50:
        z_range = midline[:, 2].max() - midline[:, 2].min()
        z_mid = midline[:, 2].min() + z_range * 0.55
        face_mask = (midline[:, 2] > z_mid - 15) & (midline[:, 2] < z_mid + 15)
        face_pts = midline[face_mask]
        if len(face_pts) > 10:
            ant_10pct = np.percentile(face_pts[:, 1], 5)
            ant_mask = face_pts[:, 1] < ant_10pct + 5
            ant_pts = face_pts[ant_mask]
            if len(ant_pts) > 0:
                lm["N"] = ant_pts[ant_pts[:, 2].argmax()].tolist()

    # -- ANS (Anterior Nasal Spine): most anterior midline point -----
    # below N, at the level of the nasal aperture inferior edge
    if len(midline) > 50 and "N" in lm:
        z_n = lm["N"][2]
        ans_zone = midline[(midline[:, 2] > z_n - 30) & (midline[:, 2] < z_n - 5)]
        if len(ans_zone) > 5:
            lm["ANS"] = ans_zone[ans_zone[:, 1].argmin()].tolist()

    # -- PNS (Posterior Nasal Spine): most posterior midline palate ---
    if "ANS" in lm:
        z_ans = lm["ANS"][2]
        pns_zone = midline[(midline[:, 2] > z_ans - 5) & (midline[:, 2] < z_ans + 5)]
        if len(pns_zone) > 5:
            lm["PNS"] = pns_zone[pns_zone[:, 1].argmax()].tolist()

    # -- A point: deepest concavity between ANS and nasal root -------
    if "ANS" in lm and "N" in lm:
        z_ans = lm["ANS"][2]
        z_n = lm["N"][2]
        a_zone = midline[(midline[:, 2] > z_ans) & (midline[:, 2] < z_n)]
        if len(a_zone) > 5:
            lm["A"] = a_zone[a_zone[:, 1].argmax()].tolist()

    # -- Orbital landmarks (per side) --------------------------------
    for side_label, x_sign in [("R", -1), ("L", 1)]:
        if x_sign == -1:
            side_mask = phys[:, 0] < x_center - 15
            side = phys[side_mask]
        else:
            side_mask = phys[:, 0] > x_center + 15
            side = phys[side_mask]

        if len(side) < 100:
            continue

        # Orbital region: anterior, roughly middle-upper height
        z_range = side[:, 2].max() - side[:, 2].min()
        z_orbit = side[:, 2].min() + z_range * 0.55
        orbit_mask = (
            (side[:, 2] > z_orbit - 15) &
            (side[:, 2] < z_orbit + 15) &
            (np.abs(side[:, 0] - x_center) < 45) &
            (np.abs(side[:, 0] - x_center) > 15)
        )
        orbit = side[orbit_mask]

        if len(orbit) > 20:
            # InfOr — most inferior point of orbital rim (anterior part)
            ant_orbit = orbit[orbit[:, 1] < np.percentile(orbit[:, 1], 30)]
            if len(ant_orbit) > 5:
                lm[f"{side_label}InfOr"] = ant_orbit[ant_orbit[:, 2].argmin()].tolist()

            # Or — lateral orbital rim, roughly at mid-orbit height
            z_orb_mid = np.median(orbit[:, 2])
            or_mask = np.abs(orbit[:, 2] - z_orb_mid) < 5
            or_pts = orbit[or_mask]
            if len(or_pts) > 5:
                if x_sign == -1:
                    lm[f"{side_label}Or"] = or_pts[or_pts[:, 0].argmin()].tolist()
                else:
                    lm[f"{side_label}Or"] = or_pts[or_pts[:, 0].argmax()].tolist()

        # MZyg — mid-zygomatic arch: most lateral point at zygomatic level
        zyg_mask = (
            (side[:, 2] > z_orbit - 20) &
            (side[:, 2] < z_orbit) &
            (np.abs(side[:, 0] - x_center) > 30)
        )
        zyg = side[zyg_mask]
        if len(zyg) > 10:
            if x_sign == -1:
                lm[f"{side_label}MZyg"] = zyg[zyg[:, 0].argmin()].tolist()
            else:
                lm[f"{side_label}MZyg"] = zyg[zyg[:, 0].argmax()].tolist()

        # FZyg — frontozygomatic suture: most lateral point of lateral orbital rim
        fzyg_mask = (
            (side[:, 2] > z_orbit) &
            (side[:, 2] < z_orbit + 15) &
            (np.abs(side[:, 0] - x_center) > 25)
        )
        fzyg = side[fzyg_mask]
        if len(fzyg) > 5:
            if x_sign == -1:
                lm[f"{side_label}FZyg"] = fzyg[fzyg[:, 0].argmin()].tolist()
            else:
                lm[f"{side_label}FZyg"] = fzyg[fzyg[:, 0].argmax()].tolist()

    # -- Po (Porion): near external auditory meatus ------------------
    # Most lateral point at the level just above the ear canal
    for side_label, x_sign in [("R", -1), ("L", 1)]:
        if x_sign == -1:
            lat_mask = phys[:, 0] < x_center - 50
        else:
            lat_mask = phys[:, 0] > x_center + 50

        lat = phys[lat_mask]
        if len(lat) < 50:
            continue

        z_range = lat[:, 2].max() - lat[:, 2].min()
        z_po = lat[:, 2].min() + z_range * 0.45
        po_mask = (lat[:, 2] > z_po - 10) & (lat[:, 2] < z_po + 10)
        po_pts = lat[po_mask]
        if len(po_pts) > 5:
            if x_sign == -1:
                lm[f"{side_label}Po"] = po_pts[po_pts[:, 0].argmin()].tolist()
            else:
                lm[f"{side_label}Po"] = po_pts[po_pts[:, 0].argmax()].tolist()

    return lm


# =====================================================================
# Landmark extraction — vertebrae
# =====================================================================

def extract_vertebra_landmark(mask_path: str, label: str) -> dict[str, list[float]]:
    """Centroid of the anterior aspect of the vertebral body."""
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    coords = np.argwhere(arr > 0)

    if len(coords) < 50:
        return {}

    origin, M = build_affine(img)
    phys = voxels_to_physical(coords, origin, M)

    ant_mask = phys[:, 1] < np.percentile(phys[:, 1], 30)
    ant = phys[ant_mask]

    if len(ant) < 10:
        return {}

    centroid = ant.mean(axis=0)
    return {label: centroid.tolist()}


# =====================================================================
# .mrk.json writer
# =====================================================================

def create_mrk_json(landmarks: dict[str, list[float]], out_path: str):
    cps = []
    for idx, (label, pos) in enumerate(landmarks.items(), 1):
        cps.append({
            "id": str(idx),
            "label": label,
            "description": "",
            "associatedNodeID": "",
            "position": pos,
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": True,
            "visibility": True,
            "positionStatus": "defined",
        })

    data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [{
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "locked": False,
            "labelFormat": "%N-%d",
            "controlPoints": cps,
            "measurements": [],
            "display": {
                "visibility": False, "opacity": 1.0,
                "color": [0.5, 0.5, 0.5],
                "selectedColor": [0.267, 0.675, 0.392],
                "propertiesLabelVisibility": False,
                "pointLabelsVisibility": True,
                "textScale": 2.0, "glyphType": "Sphere3D",
                "glyphScale": 2.0, "glyphSize": 5.0,
                "useGlyphScale": True, "sliceProjection": False,
                "sliceProjectionUseFiducialColor": True,
                "sliceProjectionOutlinedBehindSlicePlane": False,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0, "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": False,
                "snapMode": "toVisibleSurface",
            },
        }],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# =====================================================================
# Image resampling & contrast correction
# =====================================================================

def resample_image(image: sitk.Image, target_spacing: list[float]) -> sitk.Image:
    orig_sp = np.array(image.GetSpacing())
    orig_sz = np.array(image.GetSize())

    new_sz = (orig_sz * orig_sp / target_spacing).astype(int).tolist()
    out_phys = np.array(new_sz) * np.array(target_spacing)
    in_phys = orig_sz * orig_sp
    new_origin = np.array(image.GetOrigin()) - (out_phys - in_phys) / 2.0

    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(target_spacing)
    rs.SetSize(new_sz)
    rs.SetOutputDirection(image.GetDirection())
    rs.SetOutputOrigin(new_origin.tolist())
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(-1000)
    return rs.Execute(image)


def correct_histo(image: sitk.Image, min_pct=0.01, max_pct=0.95,
                  i_min=-1500, i_max=4000) -> sitk.Image:
    img = sitk.Cast(image, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)

    lo, hi = float(arr.min()), float(arr.max())
    rng = hi - lo
    histo, _ = np.histogram(arr.ravel(), bins=1000)
    cum = np.cumsum(histo).astype(float)
    cum = (cum - cum.min()) / (cum.max() - cum.min())

    lo_bin = np.searchsorted(cum, min_pct)
    hi_bin = np.searchsorted(cum, max_pct)
    res_min = max(lo_bin * rng / 1000 + lo, i_min)
    res_max = min(hi_bin * rng / 1000 + lo, i_max)

    arr = np.clip(arr, res_min, res_max)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return sitk.Cast(out, sitk.sitkInt16)


# =====================================================================
# Case discovery
# =====================================================================

SCAN_EXTS = (".nii.gz", ".nii", ".nrrd", ".mha")
SEG_DIR_SUFFIXES = ("_seg", "_segmentations", "/segmentations", "/segs")


def find_seg_dir(scan_path: str) -> str | None:
    """Try common TotalSegmentator output folder locations."""
    base = scan_path
    for ext in SCAN_EXTS:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break

    candidates = [base + sfx for sfx in SEG_DIR_SUFFIXES]
    candidates.append(os.path.join(os.path.dirname(scan_path), "segmentations"))

    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def discover_cases_data_dir(data_dir: str) -> list[tuple[str, str, str]]:
    """Auto-discover (case_id, scan_path, seg_dir) from a flat data_dir."""
    cases = []
    seen_bases = set()

    for f in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, f)
        if not os.path.isfile(full):
            continue
        if not any(f.endswith(e) for e in SCAN_EXTS):
            continue

        seg_dir = find_seg_dir(full)
        if seg_dir is None:
            continue

        case_id = f
        for ext in SCAN_EXTS:
            if case_id.endswith(ext):
                case_id = case_id[: -len(ext)]
                break

        if case_id in seen_bases:
            continue
        seen_bases.add(case_id)
        cases.append((case_id, full, seg_dir))

    return cases


def discover_cases_split(scans_dir: str, segs_dir: str) -> list[tuple[str, str, str]]:
    """Discover cases from separate scan and segmentation directories."""
    cases = []
    for f in sorted(os.listdir(scans_dir)):
        full = os.path.join(scans_dir, f)
        if not os.path.isfile(full):
            continue
        if not any(f.endswith(e) for e in SCAN_EXTS):
            continue

        case_id = f
        for ext in SCAN_EXTS:
            if case_id.endswith(ext):
                case_id = case_id[: -len(ext)]
                break

        seg_dir = os.path.join(segs_dir, case_id)
        if not os.path.isdir(seg_dir):
            print(f"  SKIP {case_id}: no segmentation dir at {seg_dir}")
            continue

        cases.append((case_id, full, seg_dir))

    return cases


# =====================================================================
# Spacing key for filenames (1.0 → "1", 0.3 → "0-3")
# =====================================================================

def spacing_key(sp: float) -> str:
    if sp == int(sp):
        return str(int(sp))
    return str(sp).replace(".", "-")


# =====================================================================
# Process one case
# =====================================================================

def process_case(
    case_id: str,
    scan_path: str,
    seg_dir: str,
    output_dir: str,
    spacings: list[float],
    do_histo: bool,
) -> dict | None:
    print(f"  [{case_id}]")

    # --- Check available masks --------------------------------------
    masks = {}
    for key, fname in {**REQUIRED_MASKS, **OPTIONAL_MASKS}.items():
        p = os.path.join(seg_dir, fname)
        if os.path.isfile(p):
            masks[key] = p

    missing = [k for k in REQUIRED_MASKS if k not in masks]
    if missing:
        print(f"    SKIP — missing masks: {missing}")
        return None

    # --- Extract landmarks ------------------------------------------
    all_lm: dict[str, list[float]] = {}

    print(f"    extracting mandible landmarks …")
    all_lm.update(extract_mandible_landmarks(masks["mandible"]))

    print(f"    extracting skull landmarks …")
    all_lm.update(extract_skull_landmarks(masks["skull"]))

    for v_key in ("vertebrae_C2", "vertebrae_C3", "vertebrae_C4"):
        if v_key in masks:
            label = v_key.replace("vertebrae_", "")
            all_lm.update(extract_vertebra_landmark(masks[v_key], label))

    if not all_lm:
        print(f"    SKIP — 0 landmarks extracted")
        return None

    print(f"    extracted {len(all_lm)} landmarks")

    # --- Save scans at each spacing ---------------------------------
    case_dir = os.path.join(output_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)

    scan = sitk.ReadImage(scan_path)
    if do_histo:
        scan = correct_histo(scan)

    current_sp = np.array(scan.GetSpacing())
    for sp_val in spacings:
        sp_str = spacing_key(sp_val)
        out_path = os.path.join(case_dir, f"{case_id}_scan_sp{sp_str}.nii.gz")
        if np.allclose(current_sp, sp_val, atol=0.01):
            sitk.WriteImage(scan, out_path)
        else:
            sitk.WriteImage(resample_image(scan, [sp_val] * 3), out_path)
        print(f"    saved scan sp{sp_str}")

    # --- Group and save landmarks -----------------------------------
    groups: dict[str, dict] = defaultdict(dict)
    for name, pos in all_lm.items():
        grp = LANDMARK_GROUPS.get(name, "CB")
        groups[grp][name] = pos

    for grp, lms in sorted(groups.items()):
        jp = os.path.join(case_dir, f"{case_id}_lm_{grp}.mrk.json")
        create_mrk_json(lms, jp)
        print(f"    saved {len(lms)} {grp}-landmarks")

    return {
        "case_id": case_id,
        "landmarks": sorted(all_lm.keys()),
        "counts": {g: len(l) for g, l in groups.items()},
    }


# =====================================================================
# Main
# =====================================================================

def main(args):
    print("=" * 60)
    print("  TotalSegmentator  →  ALI_CBCT  dataset converter")
    print("  (bone landmarks only, no teeth)")
    print("=" * 60)

    # Discover cases
    if args.scans_dir and args.segs_dir:
        cases = discover_cases_split(args.scans_dir, args.segs_dir)
    elif args.data_dir:
        cases = discover_cases_data_dir(args.data_dir)
    else:
        sys.exit("Provide --data_dir or both --scans_dir and --segs_dir")

    print(f"  Found {len(cases)} cases")
    print(f"  Spacings: {args.spacings}")
    print()

    if not cases:
        sys.exit("No cases found.")

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    all_lm_seen: set[str] = set()
    t0 = time.time()

    for i, (cid, scan_p, seg_d) in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}]")
        try:
            info = process_case(cid, scan_p, seg_d,
                                args.output_dir, args.spacings, args.correct_histo)
        except Exception as e:
            print(f"  ERROR: {e}")
            info = None

        if info:
            results.append(info)
            all_lm_seen.update(info["landmarks"])

    elapsed = time.time() - t0

    # --- Report -----------------------------------------------------
    report = {
        "total_cases": len(cases),
        "converted": len(results),
        "spacings": args.spacings,
        "all_landmarks": sorted(all_lm_seen),
        "cases": results,
    }
    rp = os.path.join(args.output_dir, "conversion_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    by_group: dict[str, list[str]] = defaultdict(list)
    for lm in sorted(all_lm_seen):
        by_group[LANDMARK_GROUPS.get(lm, "CB")].append(lm)

    print()
    print("=" * 60)
    print(f"  Done in {elapsed:.0f}s")
    print(f"  Converted: {len(results)}/{len(cases)} cases")
    print(f"  Total unique landmarks: {len(all_lm_seen)}")
    for g in ("CB", "U", "L"):
        lms = by_group.get(g, [])
        print(f"    {g}: {len(lms)}  {lms}")
    print()
    print(f"  Report: {rp}")
    print()
    print("  Paste into GlobalVar.py → LABELS_TO_TRAIN:")
    print(f"    LABELS_TO_TRAIN = {sorted(all_lm_seen)}")
    print()
    print("  Then run training:")
    sp_str = " ".join(str(s) for s in args.spacings)
    print(f"    python train_ALI_agent.py \\")
    print(f"        --dir_scans {args.output_dir} \\")
    print(f"        --scale_spacing {sp_str}")
    print()
    print("  TIP: Verify landmarks in 3D Slicer before training!")
    print("       Open any _lm_CB.mrk.json alongside the scan.")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="TotalSegmentator → ALI_CBCT (bone landmarks)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inp = p.add_argument_group("input (pick one layout)")
    inp.add_argument("--data_dir",
                     help="Directory with scans and *_seg/ subfolders side-by-side")
    inp.add_argument("--scans_dir",
                     help="Directory with scan files (use with --segs_dir)")
    inp.add_argument("--segs_dir",
                     help="Directory with per-patient TotalSegmentator folders")

    out = p.add_argument_group("output")
    out.add_argument("--output_dir", required=True,
                     help="Output directory for ALI_CBCT-formatted data")
    out.add_argument("--spacings", nargs="+", type=float, default=[1.0, 0.3],
                     help="Target spacings (mm)")

    out.add_argument("--correct_histo", action="store_true", default=True)
    out.add_argument("--no_correct_histo", dest="correct_histo",
                     action="store_false")

    args = p.parse_args()
    main(args)
