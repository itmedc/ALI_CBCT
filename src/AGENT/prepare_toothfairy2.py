#!/usr/bin/env python3
"""
Convert ToothFairy2 CBCT dataset to ALI_CBCT training format.

Reads ToothFairy2 segmentation masks and extracts dental landmark coordinates
suitable for training ALI_CBCT models. Produces NIfTI scans at two spacings
and .mrk.json landmark files grouped by Upper/Lower.

Usage:
    python prepare_toothfairy2.py \
        --input_dir  /path/to/Dataset_ToothFairy2 \
        --output_dir /path/to/output_patients

Expected input (nnU-Net format):
    Dataset_ToothFairy2/
    ├── dataset.json
    ├── imagesTr/
    │   ├── P001_0000.mha
    │   └── ...
    └── labelsTr/
        ├── P001.mha
        └── ...

Output (ALI_CBCT format):
    output_patients/
    ├── P001/
    │   ├── P001_scan_sp1.nii.gz
    │   ├── P001_scan_sp0-3.nii.gz
    │   ├── P001_lm_U.mrk.json
    │   └── P001_lm_L.mrk.json
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
# FDI → ALI_CBCT mapping
# =====================================================================

FDI_TO_ALI = {
    11: "UR1", 12: "UR2", 13: "UR3", 14: "UR4",
    15: "UR5", 16: "UR6", 17: "UR7",
    21: "UL1", 22: "UL2", 23: "UL3", 24: "UL4",
    25: "UL5", 26: "UL6", 27: "UL7",
    31: "LL1", 32: "LL2", 33: "LL3", 34: "LL4",
    35: "LL5", 36: "LL6", 37: "LL7",
    41: "LR1", 42: "LR2", 43: "LR3", 44: "LR4",
    45: "LR5", 46: "LR6", 47: "LR7",
}

# ToothFairy2 default label scheme (FDI-extended)
DEFAULT_MANDIBLE_LABEL = 1
DEFAULT_MAXILLA_LABEL = 2

# ALI_CBCT landmark → group assignment
UPPER_GROUP_NAMES = {
    "RInfOr", "LInfOr", "LMZyg", "RPF", "LPF", "PNS", "ANS", "A",
    "IF", "ROr", "LOr", "RMZyg", "RNC", "LNC",
}
LOWER_GROUP_NAMES = {
    "RCo", "RGo", "Me", "Gn", "Pog", "PogL", "B", "LGo", "LCo",
    "LAF", "LAE", "RAF", "RAE",
}


def landmark_group(name: str) -> str:
    if name in UPPER_GROUP_NAMES:
        return "U"
    if name in LOWER_GROUP_NAMES:
        return "L"
    if name.startswith("U"):
        return "U"
    if name.startswith("L") and len(name) >= 3:
        return "L"
    if name in {"Me", "Gn", "Pog", "PogL", "B", "RGo", "LGo", "RCo", "LCo"}:
        return "L"
    return "U"


# =====================================================================
# Affine helpers
# =====================================================================

def build_voxel_to_physical(image: sitk.Image):
    """Return (origin, M) so that physical = origin + M @ [x, y, z]^T."""
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    M = direction @ np.diag(spacing)
    return origin, M


def voxels_to_physical(voxels_zyx: np.ndarray, origin: np.ndarray, M: np.ndarray):
    """Convert (N, 3) voxel coords in [z, y, x] order to (N, 3) LPS physical."""
    xyz = voxels_zyx[:, ::-1].astype(np.float64)
    return (origin[None, :] + xyz @ M.T)


def centroid_physical(voxels_zyx: np.ndarray, origin: np.ndarray, M: np.ndarray):
    """Centroid of voxels in physical coordinates."""
    mean_zyx = voxels_zyx.mean(axis=0, keepdims=True)
    return voxels_to_physical(mean_zyx, origin, M)[0]


# =====================================================================
# Landmark extraction — teeth
# =====================================================================

def extract_tooth_landmarks(
    seg_array: np.ndarray,
    origin: np.ndarray,
    M: np.ndarray,
    fdi_id: int,
    ali_prefix: str,
    is_upper: bool,
) -> dict:
    """
    Extract O (occlusal/crown tip) and R (root/apex) landmarks for one tooth.
    For first molars also extracts MB and DB cusps.
    """
    mask = seg_array == fdi_id
    if not np.any(mask):
        return {}

    coords = np.argwhere(mask)  # (N,3) in [z,y,x]
    if len(coords) < 10:
        return {}

    phys = voxels_to_physical(coords, origin, M)
    z_phys = phys[:, 2]  # S-axis in LPS

    landmarks = {}

    # -- Occlusal (O) and Root (R) -----------------------------------
    if is_upper:
        occ_slice_mask = z_phys <= np.percentile(z_phys, 3)
        root_slice_mask = z_phys >= np.percentile(z_phys, 97)
    else:
        occ_slice_mask = z_phys >= np.percentile(z_phys, 97)
        root_slice_mask = z_phys <= np.percentile(z_phys, 3)

    occ_centroid = phys[occ_slice_mask].mean(axis=0)
    root_centroid = phys[root_slice_mask].mean(axis=0)

    landmarks[f"{ali_prefix}O"] = occ_centroid.tolist()
    landmarks[f"{ali_prefix}R"] = root_centroid.tolist()

    # -- Molar cusps (MB / DB) for tooth 6 ---------------------------
    tooth_pos = fdi_id % 10
    if tooth_pos == 6:
        mb, db = _extract_molar_cusps(phys, z_phys, ali_prefix, is_upper)
        landmarks.update(mb)
        landmarks.update(db)

    return landmarks


def _extract_molar_cusps(
    phys: np.ndarray,
    z_phys: np.ndarray,
    ali_prefix: str,
    is_upper: bool,
) -> tuple[dict, dict]:
    """Split crown into mesial/distal halves; find cusp tip in each."""
    mb, db = {}, {}

    if is_upper:
        crown_mask = z_phys <= np.percentile(z_phys, 25)
    else:
        crown_mask = z_phys >= np.percentile(z_phys, 75)

    crown = phys[crown_mask]
    crown_z = z_phys[crown_mask]

    if len(crown) < 20:
        return mb, db

    y_median = np.median(crown[:, 1])  # P-axis: higher = more posterior

    mesial = crown[crown[:, 1] < y_median]
    distal = crown[crown[:, 1] >= y_median]

    if len(mesial) > 5:
        if is_upper:
            tip = mesial[mesial[:, 2].argmin()]
        else:
            tip = mesial[mesial[:, 2].argmax()]
        mb[f"{ali_prefix}MB"] = tip.tolist()

    if len(distal) > 5:
        if is_upper:
            tip = distal[distal[:, 2].argmin()]
        else:
            tip = distal[distal[:, 2].argmax()]
        db[f"{ali_prefix}DB"] = tip.tolist()

    return mb, db


# =====================================================================
# Landmark extraction — mandible
# =====================================================================

def extract_mandible_landmarks(
    seg_array: np.ndarray,
    origin: np.ndarray,
    M: np.ndarray,
    mandible_label: int,
) -> dict:
    """
    Approximate key mandibular landmarks from the mandible segmentation mask:
    Me, Gn, Pog, PogL, B, RGo, LGo, RCo, LCo.
    """
    mask = seg_array == mandible_label
    if not np.any(mask):
        return {}

    coords = np.argwhere(mask)

    if len(coords) > 80_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(coords), 80_000, replace=False)
        coords = coords[idx]

    phys = voxels_to_physical(coords, origin, M)
    landmarks = {}

    x_center = np.median(phys[:, 0])
    midline_tol = 3.0  # mm

    midline_mask = np.abs(phys[:, 0] - x_center) < midline_tol
    midline = phys[midline_mask]

    if len(midline) < 20:
        return landmarks

    # Me (Menton): most inferior midline point
    me_idx = midline[:, 2].argmin()
    landmarks["Me"] = midline[me_idx].tolist()

    # Pog (Pogonion): most anterior midline point in chin region
    z_me = midline[me_idx, 2]
    chin_mask = midline[:, 2] < z_me + 10
    chin = midline[chin_mask]
    if len(chin) > 0:
        pog_idx = chin[:, 1].argmin()  # most anterior = min P
        landmarks["Pog"] = chin[pog_idx].tolist()
        landmarks["PogL"] = chin[pog_idx].tolist()

    # Gn (Gnathion): midpoint Me–Pog
    if "Me" in landmarks and "Pog" in landmarks:
        gn = [(landmarks["Me"][i] + landmarks["Pog"][i]) / 2 for i in range(3)]
        landmarks["Gn"] = gn

    # B point: most posterior midline point between Pog and 20 mm above it
    if "Pog" in landmarks:
        z_pog = landmarks["Pog"][2]
        b_mask = (midline[:, 2] > z_pog + 5) & (midline[:, 2] < z_pog + 20)
        b_region = midline[b_mask]
        if len(b_region) > 0:
            b_idx = b_region[:, 1].argmax()  # most posterior = max P
            landmarks["B"] = b_region[b_idx].tolist()

    # RGo / LGo / RCo / LCo — lateral landmarks
    for side, name_prefix in [("R", -1), ("L", 1)]:
        if name_prefix == -1:
            side_mask = phys[:, 0] < x_center - 20
        else:
            side_mask = phys[:, 0] > x_center + 20

        side_pts = phys[side_mask]
        if len(side_pts) < 50:
            continue

        # Go = most inferior-posterior point of the mandibular angle
        go_score = side_pts[:, 1] - side_pts[:, 2]  # max posterior + min z
        go_idx = go_score.argmax()
        landmarks[f"{side}Go"] = side_pts[go_idx].tolist()

        # Co = most superior point of the condylar head (posterior ramus)
        post_mask = side_pts[:, 1] > np.percentile(side_pts[:, 1], 75)
        ramus_pts = side_pts[post_mask]
        if len(ramus_pts) > 0:
            co_idx = ramus_pts[:, 2].argmax()
            landmarks[f"{side}Co"] = ramus_pts[co_idx].tolist()

    return landmarks


# =====================================================================
# .mrk.json writer (3D Slicer Markups)
# =====================================================================

def create_mrk_json(landmarks_dict: dict, out_path: str):
    control_points = []
    for idx, (label, position) in enumerate(landmarks_dict.items(), 1):
        control_points.append({
            "id": str(idx),
            "label": label,
            "description": "",
            "associatedNodeID": "",
            "position": position,
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
            "controlPoints": control_points,
            "measurements": [],
            "display": {
                "visibility": False,
                "opacity": 1.0,
                "color": [0.5, 0.5, 0.5],
                "selectedColor": [0.267, 0.675, 0.392],
                "propertiesLabelVisibility": False,
                "pointLabelsVisibility": True,
                "textScale": 2.0,
                "glyphType": "Sphere3D",
                "glyphScale": 2.0,
                "glyphSize": 5.0,
                "useGlyphScale": True,
                "sliceProjection": False,
                "sliceProjectionUseFiducialColor": True,
                "sliceProjectionOutlinedBehindSlicePlane": False,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
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
# Image resampling
# =====================================================================

def resample_image(image: sitk.Image, target_spacing: list[float]) -> sitk.Image:
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    new_size = (original_size * original_spacing / target_spacing).astype(int).tolist()

    output_physical_size = np.array(new_size) * np.array(target_spacing)
    input_physical_size = original_size * original_spacing
    new_origin = np.array(image.GetOrigin()) - (output_physical_size - input_physical_size) / 2.0

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(new_origin.tolist())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)

    return resampler.Execute(image)


# =====================================================================
# Contrast correction (matches ALI_CBCT CorrectHisto)
# =====================================================================

def correct_histo(image: sitk.Image, min_pct=0.01, max_pct=0.95,
                  i_min=-1500, i_max=4000) -> sitk.Image:
    img = sitk.Cast(image, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)

    img_min, img_max = float(arr.min()), float(arr.max())
    img_range = img_max - img_min

    histo, _ = np.histogram(arr.ravel(), bins=1000)
    cum = np.cumsum(histo).astype(float)
    cum = (cum - cum.min()) / (cum.max() - cum.min())

    res_low_bin = np.searchsorted(cum, min_pct)
    res_high_bin = np.searchsorted(cum, max_pct)
    res_min = max(res_low_bin * img_range / 1000 + img_min, i_min)
    res_max = min(res_high_bin * img_range / 1000 + img_min, i_max)

    arr = np.clip(arr, res_min, res_max)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return sitk.Cast(out, sitk.sitkInt16)


# =====================================================================
# dataset.json parsing
# =====================================================================

def load_label_mapping(input_dir: str) -> dict[int, str]:
    """Return {label_id: name} from dataset.json, or empty dict."""
    p = os.path.join(input_dir, "dataset.json")
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        dj = json.load(f)
    labels = dj.get("labels", {})
    return {int(k): str(v) for k, v in labels.items()}


def detect_mandible_label(label_map: dict[int, str]) -> int:
    for lid, name in label_map.items():
        if "mandible" in name.lower():
            return lid
    return DEFAULT_MANDIBLE_LABEL


def detect_fdi_labels(label_map: dict[int, str]) -> dict[int, str]:
    """Return {fdi_id: ali_prefix} for teeth present in label_map."""
    if not label_map:
        return dict(FDI_TO_ALI)

    result = {}
    for lid, name in label_map.items():
        if lid in FDI_TO_ALI:
            result[lid] = FDI_TO_ALI[lid]
    return result if result else dict(FDI_TO_ALI)


# =====================================================================
# Case processing
# =====================================================================

def spacing_key(sp: float) -> str:
    """1.0 → '1', 0.3 → '0-3'."""
    if sp == int(sp):
        return str(int(sp))
    return str(sp).replace(".", "-")


def process_case(
    case_id: str,
    scan_path: str,
    seg_path: str,
    output_dir: str,
    fdi_labels: dict[int, str],
    mandible_label: int,
    spacings: list[float],
    do_correct_histo: bool,
) -> dict | None:
    print(f"  [{case_id}] reading images …")
    scan = sitk.ReadImage(scan_path)
    seg = sitk.ReadImage(seg_path)
    seg_array = sitk.GetArrayFromImage(seg)

    unique_labels = set(np.unique(seg_array).astype(int))
    available_fdi = {fdi: ali for fdi, ali in fdi_labels.items()
                     if fdi in unique_labels}

    if not available_fdi and mandible_label not in unique_labels:
        print(f"  [{case_id}] SKIP — no usable structures")
        return None

    origin, M = build_voxel_to_physical(seg)

    # --- Extract landmarks ------------------------------------------
    all_lm: dict[str, list[float]] = {}

    for fdi_id, ali_prefix in available_fdi.items():
        is_upper = fdi_id < 30
        lms = extract_tooth_landmarks(seg_array, origin, M,
                                      fdi_id, ali_prefix, is_upper)
        all_lm.update(lms)

    if mandible_label in unique_labels:
        lms = extract_mandible_landmarks(seg_array, origin, M, mandible_label)
        all_lm.update(lms)

    if not all_lm:
        print(f"  [{case_id}] SKIP — 0 landmarks extracted")
        return None

    # --- Save scans at each spacing ---------------------------------
    case_dir = os.path.join(output_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)

    if do_correct_histo:
        scan = correct_histo(scan)

    current_sp = np.array(scan.GetSpacing())

    for sp_val in spacings:
        sp_str = spacing_key(sp_val)
        out_path = os.path.join(case_dir, f"{case_id}_scan_sp{sp_str}.nii.gz")

        if np.allclose(current_sp, sp_val, atol=0.01):
            sitk.WriteImage(scan, out_path)
        else:
            resampled = resample_image(scan, [sp_val] * 3)
            sitk.WriteImage(resampled, out_path)

        print(f"  [{case_id}] saved scan sp{sp_str}")

    # --- Group landmarks U / L and save .mrk.json -------------------
    groups: dict[str, dict] = defaultdict(dict)
    for name, pos in all_lm.items():
        groups[landmark_group(name)][name] = pos

    for grp, lms in groups.items():
        json_path = os.path.join(case_dir, f"{case_id}_lm_{grp}.mrk.json")
        create_mrk_json(lms, json_path)
        print(f"  [{case_id}] saved {len(lms)} {grp}-landmarks")

    return {
        "case_id": case_id,
        "teeth_found": sorted(available_fdi.values()),
        "landmarks": sorted(all_lm.keys()),
        "upper": len(groups.get("U", {})),
        "lower": len(groups.get("L", {})),
    }


# =====================================================================
# Discover cases
# =====================================================================

def discover_cases(input_dir: str) -> list[tuple[str, str, str]]:
    images_dir = os.path.join(input_dir, "imagesTr")
    labels_dir = os.path.join(input_dir, "labelsTr")

    if not os.path.isdir(images_dir):
        sys.exit(f"ERROR: imagesTr not found at {images_dir}")
    if not os.path.isdir(labels_dir):
        sys.exit(f"ERROR: labelsTr not found at {labels_dir}")

    cases = []
    for fname in sorted(os.listdir(labels_dir)):
        stem = fname.split(".")[0]
        seg_path = os.path.join(labels_dir, fname)
        if not os.path.isfile(seg_path):
            continue

        scan_candidates = [
            os.path.join(images_dir, f"{stem}_0000.mha"),
            os.path.join(images_dir, f"{stem}_0000.nii.gz"),
            os.path.join(images_dir, f"{stem}_0000.nrrd"),
        ]
        scan_path = next((p for p in scan_candidates if os.path.isfile(p)), None)
        if scan_path is None:
            print(f"  WARNING: no scan found for {stem}, skipping")
            continue

        cases.append((stem, scan_path, seg_path))

    return cases


# =====================================================================
# Main
# =====================================================================

def main(args):
    print("=" * 60)
    print("  ToothFairy2  →  ALI_CBCT  dataset converter")
    print("=" * 60)
    print(f"  Input:   {args.input_dir}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Spacings: {args.spacings}")
    print()

    label_map = load_label_mapping(args.input_dir)
    if label_map:
        print(f"  Loaded dataset.json with {len(label_map)} labels")
    else:
        print("  dataset.json not found — using default FDI label IDs")

    fdi_labels = detect_fdi_labels(label_map)
    mandible_label = detect_mandible_label(label_map)
    print(f"  Detected {len(fdi_labels)} tooth labels, mandible={mandible_label}")

    cases = discover_cases(args.input_dir)
    print(f"  Found {len(cases)} cases\n")

    if not cases:
        sys.exit("No cases found.")

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    all_landmarks_seen: set[str] = set()
    t0 = time.time()

    for i, (case_id, scan_p, seg_p) in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case_id}")
        try:
            info = process_case(
                case_id, scan_p, seg_p,
                args.output_dir, fdi_labels, mandible_label,
                args.spacings, args.correct_histo,
            )
        except Exception as e:
            print(f"  ERROR on {case_id}: {e}")
            info = None

        if info:
            results.append(info)
            all_landmarks_seen.update(info["landmarks"])

    elapsed = time.time() - t0

    # --- Summary report ---------------------------------------------
    report = {
        "total_cases": len(cases),
        "converted_cases": len(results),
        "spacings": args.spacings,
        "all_landmarks": sorted(all_landmarks_seen),
        "cases": results,
    }
    report_path = os.path.join(args.output_dir, "conversion_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    upper_lms = sorted(lm for lm in all_landmarks_seen if landmark_group(lm) == "U")
    lower_lms = sorted(lm for lm in all_landmarks_seen if landmark_group(lm) == "L")

    print()
    print("=" * 60)
    print(f"  Done in {elapsed:.0f}s")
    print(f"  Converted: {len(results)}/{len(cases)} cases")
    print(f"  Total unique landmarks: {len(all_landmarks_seen)}")
    print(f"    Upper (U): {len(upper_lms)}")
    print(f"    Lower (L): {len(lower_lms)}")
    print(f"  Report: {report_path}")
    print()
    print("  Recommended LABELS_TO_TRAIN for GlobalVar.py:")
    print(f"    Upper teeth: {upper_lms[:10]} ...")
    print(f"    Lower teeth: {lower_lms[:10]} ...")
    print()
    print("  Next steps:")
    print("    1) Verify landmarks in 3D Slicer (open any _lm_U.mrk.json)")
    print("    2) Copy the list above into GlobalVar.py → LABELS_TO_TRAIN")
    print("    3) Run training:")
    print(f"       python train_ALI_agent.py \\")
    print(f"           --dir_scans {args.output_dir} \\")
    print(f"           --scale_spacing {' '.join(str(s) for s in args.spacings)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ToothFairy2 → ALI_CBCT training format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Path to ToothFairy2 dataset root (with imagesTr/ and labelsTr/)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for ALI_CBCT-formatted patient data",
    )
    parser.add_argument(
        "--spacings", nargs="+", type=float, default=[1.0, 0.3],
        help="Target spacings (mm) for multi-scale scans",
    )
    parser.add_argument(
        "--correct_histo", action="store_true", default=True,
        help="Apply histogram contrast correction (recommended)",
    )
    parser.add_argument(
        "--no_correct_histo", dest="correct_histo", action="store_false",
        help="Skip histogram contrast correction",
    )
    args = parser.parse_args()
    main(args)
