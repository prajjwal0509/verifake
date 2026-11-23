"""Fit Step 4 OOD ensemble weight, bounded calibration & disagreement rules on val, then evaluate once on held_out_test."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.train_step3 import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_MANIFEST,
    STEP3_EVAL_PERTS,
    _load_clip_caches,
    infer_convnext_two_view_logits,
)
from src.augmentations import build_convnext_transforms  # noqa: E402
from src.calibration import compute_ece  # noqa: E402
from src.clip_detector import compute_fpr_threshold, load_clip_head  # noqa: E402
from src.ensemble_inference import (  # noqa: E402
    DEFAULT_BRANCH_DISAGREEMENT_MARGIN,
    DEFAULT_VIEW_DISAGREEMENT_MARGIN,
    STEP3_ENSEMBLE_PATH,
    STEP4_CALIBRATION_PATH,
    apply_calibrated_decision,
    fit_temperature_bounded_platt,
)
from src.image_io import register_image_plugins  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.utils import (  # noqa: E402
    REPORTS_DIR,
    SEED,
    get_device,
    get_logger,
    load_class_mapping,
    load_json,
    save_json,
)

logger = get_logger("calibrate_step4")

OOD_WEIGHT_GRID: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    z = np.clip(z.astype(np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logit_scalar(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _nll_np(z: np.ndarray, y: np.ndarray) -> float:
    loss_fn = torch.nn.BCEWithLogitsLoss()
    return float(
        loss_fn(
            torch.tensor(z, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ).item()
    )


@torch.no_grad()
def _score_cached_clip_views(
    head: torch.nn.Module,
    shas: list[str],
    cache: dict[str, Any],
    sha_to_idx: dict[str, int],
    *,
    pert_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (crop_logits (N,2), full_logits (N,2), face_ok (N,))."""
    pj = cache["perturbations"].index(pert_name)
    crops, fulls, oks = [], [], []
    for s in shas:
        idx = sha_to_idx[s]
        ok = bool(cache["face_detected"][idx, pj])
        slot = pj if ok else 0
        crops.append(cache["crop_feats"][idx, slot].float())
        fulls.append(cache["full_feats"][idx, slot].float())
        oks.append(ok)
    c_t = torch.stack(crops, dim=0).to(device)
    f_t = torch.stack(fulls, dim=0).to(device)
    return head(c_t).cpu(), head(f_t).cpu(), torch.tensor(oks, dtype=torch.bool)


@torch.no_grad()
def _collect_raw_branch_scalars(
    df_sub: pd.DataFrame,
    clip_head: torch.nn.Module,
    conv_model: torch.nn.Module,
    conv_tf,
    cache: dict[str, Any],
    sha_to_idx: dict[str, int],
    *,
    pert_name: str,
    fake_idx: int,
    real_idx: int,
    batch_size: int,
    device: torch.device,
    require_face: bool,
) -> dict[str, np.ndarray]:
    shas = df_sub["sha256"].astype(str).tolist()
    cl_c, cl_f, cl_ok = _score_cached_clip_views(
        clip_head, shas, cache, sha_to_idx, pert_name=pert_name, device=device
    )
    _, cv_c, cv_f, cv_ok = infer_convnext_two_view_logits(
        conv_model,
        df_sub,
        conv_tf,
        perturbation=pert_name,
        require_face=require_face,
        batch_size=batch_size,
        device=device,
    )
    return {
        "y": (df_sub["label"].to_numpy(dtype=int) == fake_idx).astype(np.int64),
        "cl_cz": (cl_c[:, fake_idx] - cl_c[:, real_idx]).numpy().astype(np.float64),
        "cl_fz": (cl_f[:, fake_idx] - cl_f[:, real_idx]).numpy().astype(np.float64),
        "cv_cz": (cv_c[:, fake_idx] - cv_c[:, real_idx]).numpy().astype(np.float64),
        "cv_fz": (cv_f[:, fake_idx] - cv_f[:, real_idx]).numpy().astype(np.float64),
        "ok": (cl_ok & cv_ok).numpy(),
    }


def _blend_scalars(raw: dict[str, np.ndarray], w_clip: float) -> dict[str, np.ndarray]:
    w_conv = 1.0 - w_clip
    crop_z = w_clip * raw["cl_cz"] + w_conv * raw["cv_cz"]
    full_z = w_clip * raw["cl_fz"] + w_conv * raw["cv_fz"]
    clip_z = 0.5 * (raw["cl_cz"] + raw["cl_fz"])
    conv_z = 0.5 * (raw["cv_cz"] + raw["cv_fz"])
    fused_z = 0.5 * (crop_z + full_z)
    return {
        "y": raw["y"],
        "crop_z": crop_z,
        "full_z": full_z,
        "clip_z": clip_z,
        "conv_z": conv_z,
        "fused_z": fused_z,
        "ok": raw["ok"],
    }


def _select_ood_clip_weight(
    val_clean_raw: dict[str, np.ndarray],
    val_clean_roles: np.ndarray,
    val_pert_raws: list[dict[str, np.ndarray]],
    val_pert_roles: np.ndarray,
) -> float:
    """Select w_clip strictly on unseen_val_generator + val_real (clean + perturbed)."""
    ood_c = np.isin(val_clean_roles, ["unseen_val_generator", "val_real"]) & val_clean_raw["ok"]
    ood_p_role = np.isin(val_pert_roles, ["unseen_val_generator", "val_real"])

    y_c = val_clean_raw["y"][ood_c]
    cl_c = 0.5 * (val_clean_raw["cl_cz"][ood_c] + val_clean_raw["cl_fz"][ood_c])
    cv_c = 0.5 * (val_clean_raw["cv_cz"][ood_c] + val_clean_raw["cv_fz"][ood_c])

    best_w = 0.55
    best_score = -1.0
    for wc in OOD_WEIGHT_GRID:
        p_c = _sigmoid_np(wc * cl_c + (1.0 - wc) * cv_c)
        thr5, _ = compute_fpr_threshold(p_c[y_c == 0], 0.05)
        c_auc = float(roc_auc_score(y_c, p_c))
        c_rec = float((p_c[y_c == 1] >= thr5).mean())

        p_aucs, p_recs = [], []
        for pr in val_pert_raws:
            pm = ood_p_role & pr["ok"]
            if pm.sum() < 20:
                continue
            y_p = pr["y"][pm]
            if not ((y_p == 0).any() and (y_p == 1).any()):
                continue
            cl_p = 0.5 * (pr["cl_cz"][pm] + pr["cl_fz"][pm])
            cv_p = 0.5 * (pr["cv_cz"][pm] + pr["cv_fz"][pm])
            p_p = _sigmoid_np(wc * cl_p + (1.0 - wc) * cv_p)
            p_aucs.append(float(roc_auc_score(y_p, p_p)))
            p_recs.append(float((p_p[y_p == 1] >= thr5).mean()))

        mean_pa = float(np.mean(p_aucs)) if p_aucs else c_auc
        mean_pr = float(np.mean(p_recs)) if p_recs else c_rec
        score = 0.5 * (0.5 * (c_auc + c_rec) + 0.5 * (mean_pa + mean_pr))
        logger.info(
            "  OOD weight sweep w_clip=%.2f -> clean AUC=%.4f Rec@5%%=%.3f | pert AUC=%.4f Rec@5%%=%.3f | score=%.4f",
            wc,
            c_auc,
            c_rec,
            mean_pa,
            mean_pr,
            score,
        )
        if score > best_score:
            best_score = score
            best_w = wc
    logger.info("Selected OOD w_clip=%.2f (OOD gen_score=%.4f)", best_w, best_score)
    return float(best_w)


def _evaluate_decisions(
    arrays: dict[str, np.ndarray],
    *,
    temperature: float,
    bias: float,
    decision_threshold: float,
    uncertain_band: tuple[float, float],
    view_margin: float,
    branch_margin: float,
) -> dict[str, Any]:
    ok = arrays["ok"]
    y = arrays["y"][ok]
    crop_z = arrays["crop_z"][ok]
    full_z = arrays["full_z"][ok]
    clip_z = arrays["clip_z"][ok]
    conv_z = arrays["conv_z"][ok]
    fused_z = arrays["fused_z"][ok]

    z_cal = (fused_z + bias) / max(1.0, temperature)
    p_uncal = _sigmoid_np(fused_z)
    p_cal = _sigmoid_np(z_cal)

    labels = []
    bin_labels = []
    uncertain_reasons = []
    view_disagrees = []
    branch_disagrees = []
    for cz, fz, clz, cvz in zip(crop_z, full_z, clip_z, conv_z):
        d = apply_calibrated_decision(
            crop_z=float(cz),
            full_z=float(fz),
            clip_z=float(clz),
            conv_z=float(cvz),
            temperature=temperature,
            bias=bias,
            decision_threshold=decision_threshold,
            uncertain_band=uncertain_band,
            view_disagreement_margin=view_margin,
            branch_disagreement_margin=branch_margin,
        )
        labels.append(d["label"])
        bin_labels.append(d["binary_label"])
        uncertain_reasons.append(d["uncertain_reason"])
        view_disagrees.append(bool(d["view_disagreement"]))
        branch_disagrees.append(bool(d["branch_disagreement"]))

    labels_arr = np.asarray(labels)
    bin_arr = np.asarray(bin_labels)
    reasons_arr = np.asarray(uncertain_reasons, dtype=object)
    vd_arr = np.asarray(view_disagrees, dtype=bool)
    bd_arr = np.asarray(branch_disagrees, dtype=bool)
    abstain_mask = labels_arr == "UNCERTAIN"
    decided_mask = ~abstain_mask

    fake_m = y == 1
    real_m = y == 0
    auroc = (
        float(roc_auc_score(y, p_cal))
        if (fake_m.any() and real_m.any())
        else None
    )

    binary_recall = float((bin_arr[fake_m] == "FAKE").mean()) if fake_m.any() else 0.0
    binary_real_fpr = float((bin_arr[real_m] == "FAKE").mean()) if real_m.any() else 0.0

    dec_fake = decided_mask & fake_m
    dec_real = decided_mask & real_m
    decided_recall = float((labels_arr[dec_fake] == "FAKE").mean()) if dec_fake.any() else 0.0
    decided_real_fpr = float((labels_arr[dec_real] == "FAKE").mean()) if dec_real.any() else 0.0

    abstained_err = (
        float(((bin_arr[abstain_mask] == "FAKE").astype(int) != y[abstain_mask]).mean())
        if abstain_mask.any()
        else 0.0
    )

    return {
        "n": int(len(y)),
        "auroc": auroc,
        "nll_before": _nll_np(fused_z, y),
        "nll_after": _nll_np(z_cal, y),
        "ece_before": compute_ece(p_uncal, y),
        "ece_after": compute_ece(p_cal, y),
        "binary_recall": binary_recall,
        "binary_real_fpr": binary_real_fpr,
        "decided_recall": decided_recall,
        "decided_real_fpr": decided_real_fpr,
        "abstention_rate": float(abstain_mask.mean()) if len(y) else 0.0,
        "n_uncertain": int(abstain_mask.sum()),
        "n_view_disagreement": int(vd_arr.sum()),
        "n_branch_disagreement": int(bd_arr.sum()),
        "n_borderline": int((reasons_arr == "borderline_confidence").sum()),
        "abstained_binary_error_rate": abstained_err,
        "p_cal": p_cal,
        "labels": labels_arr,
        "bin_labels": bin_arr,
        "view_disagrees": vd_arr,
        "branch_disagrees": bd_arr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Step 4 OOD weights + calibration and evaluate once on held_out_test.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--ensemble-json", type=Path, default=STEP3_ENSEMBLE_PATH)
    parser.add_argument("--out-cal", type=Path, default=STEP4_CALIBRATION_PATH)
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR / "step4")
    parser.add_argument("--fit-ood-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-val-fpr", type=float, default=0.04)
    parser.add_argument("--band-half-logit", type=float, default=0.55)
    parser.add_argument("--view-margin", type=float, default=DEFAULT_VIEW_DISAGREEMENT_MARGIN)
    parser.add_argument("--branch-margin", type=float, default=DEFAULT_BRANCH_DISAGREEMENT_MARGIN)
    parser.add_argument("--pert-per-source", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    register_image_plugins()
    device = get_device() if args.device == "auto" else torch.device(args.device)
    mapping = load_class_mapping()
    fake_idx = int(mapping["class_to_idx"]["FAKE"])
    real_idx = int(mapping["class_to_idx"]["REAL"])

    ens_cfg = load_json(args.ensemble_json)
    best_clip = str(ens_cfg["best_clip_branch"])
    w_clip_step3 = float(ens_cfg["ensemble"]["clip_weight"])
    clip_key = "clip_mlp_checkpoint" if best_clip == "clip_mlp" else "clip_linear_checkpoint"
    clip_head, _ = load_clip_head(PROJECT_ROOT / ens_cfg[clip_key], device)
    conv_model, _ = load_checkpoint(PROJECT_ROOT / ens_cfg["convnext_checkpoint"], device)
    conv_tf = build_convnext_transforms(train=False, img_size=224)

    _, va_cache, te_cache = _load_clip_caches(args.cache_dir)
    sha_to_va = {s: i for i, s in enumerate(va_cache["sha256"])}
    sha_to_te = {s: i for i, s in enumerate(te_cache["sha256"])}

    df = pd.read_csv(args.manifest)
    face_df = df[df["has_face"].astype(bool)].reset_index(drop=True)
    val_df = face_df[face_df["split"] == "val"].reset_index(drop=True)
    test_df = face_df[face_df["split"] == "held_out_test"].reset_index(drop=True)

    # 1. Collect raw per-branch per-view validation logits cleanly on val (clean + perturbed slice)
    val_clean_raw = _collect_raw_branch_scalars(
        val_df, clip_head, conv_model, conv_tf, va_cache, sha_to_va,
        pert_name="original", fake_idx=fake_idx, real_idx=real_idx,
        batch_size=args.batch_size, device=device, require_face=True,
    )
    val_pert_parts = [
        grp.sample(n=min(args.pert_per_source, len(grp)), random_state=args.seed + 11)
        for _, grp in val_df.groupby("generator")
    ]
    val_pert_df = pd.concat(val_pert_parts, ignore_index=True)

    val_pert_raws = []
    for p_name in STEP3_EVAL_PERTS:
        if p_name == "original":
            continue
        val_pert_raws.append(
            _collect_raw_branch_scalars(
                val_pert_df, clip_head, conv_model, conv_tf, va_cache, sha_to_va,
                pert_name=p_name, fake_idx=fake_idx, real_idx=real_idx,
                batch_size=args.batch_size, device=device, require_face=True,
            )
        )

    if args.fit_ood_weight:
        w_clip = _select_ood_clip_weight(
            val_clean_raw,
            val_df["split_role"].astype(str).to_numpy(),
            val_pert_raws,
            val_pert_df["split_role"].astype(str).to_numpy(),
        )
    else:
        w_clip = w_clip_step3

    val_clean = _blend_scalars(val_clean_raw, w_clip)
    fit_z_parts = [val_clean["fused_z"][val_clean["ok"]]]
    fit_y_parts = [val_clean["y"][val_clean["ok"]]]
    for pr in val_pert_raws:
        vp = _blend_scalars(pr, w_clip)
        fit_z_parts.append(vp["fused_z"][vp["ok"]])
        fit_y_parts.append(vp["y"][vp["ok"]])

    fit_z = np.concatenate(fit_z_parts, axis=0)
    fit_y = np.concatenate(fit_y_parts, axis=0)
    cal_fit = fit_temperature_bounded_platt(fit_z, fit_y)
    temperature = cal_fit["temperature"]
    bias = cal_fit["bias"]

    val_clean_cal_p = _sigmoid_np((val_clean["fused_z"] + bias) / temperature)
    val_real_mask = val_clean["ok"] & (val_clean["y"] == 0)
    decision_thr, actual_val_fpr = compute_fpr_threshold(
        val_clean_cal_p[val_real_mask], target_fpr=args.target_val_fpr
    )
    decision_thr = min(max(float(decision_thr), 0.01), 0.99)
    z_thr = _logit_scalar(decision_thr)
    band = [
        float(1.0 / (1.0 + math.exp(-(z_thr - args.band_half_logit)))),
        float(1.0 / (1.0 + math.exp(-(z_thr + args.band_half_logit)))),
    ]

    cal_payload = {
        "temperature": float(temperature),
        "bias": float(bias),
        "decision_threshold": float(decision_thr),
        "target_val_fpr": float(args.target_val_fpr),
        "actual_val_real_fpr": float(actual_val_fpr),
        "uncertain_band": band,
        "band_half_logit": float(args.band_half_logit),
        "view_disagreement_margin": float(args.view_margin),
        "branch_disagreement_margin": float(args.branch_margin),
        "best_clip_branch": best_clip,
        "clip_weight": float(w_clip),
        "convnext_weight": float(1.0 - w_clip),
        "ood_weight_selected": bool(args.fit_ood_weight),
        "val_n": cal_fit["val_n"],
        "val_nll_before": cal_fit["val_nll_before"],
        "val_nll_after": cal_fit["val_nll_after"],
        "val_ece_before": cal_fit["val_ece_before"],
        "val_ece_after": cal_fit["val_ece_after"],
    }
    save_json(cal_payload, args.out_cal)
    logger.info("Saved Step 4 calibration config -> %s", args.out_cal)

    # 2. Single read-only evaluation on held_out_test
    test_clean_raw = _collect_raw_branch_scalars(
        test_df, clip_head, conv_model, conv_tf, te_cache, sha_to_te,
        pert_name="original", fake_idx=fake_idx, real_idx=real_idx,
        batch_size=args.batch_size, device=device, require_face=True,
    )
    test_clean = _blend_scalars(test_clean_raw, w_clip)
    test_overall = _evaluate_decisions(
        test_clean,
        temperature=temperature,
        bias=bias,
        decision_threshold=decision_thr,
        uncertain_band=(band[0], band[1]),
        view_margin=args.view_margin,
        branch_margin=args.branch_margin,
    )

    ok = test_clean["ok"]
    df_ok = test_df[ok].reset_index(drop=True)
    p_cal = test_overall["p_cal"]
    labels = test_overall["labels"]
    bin_labels = test_overall["bin_labels"]
    vd = test_overall["view_disagrees"]
    bd = test_overall["branch_disagrees"]

    real_scores_all = p_cal[df_ok["label"].to_numpy(dtype=int) == real_idx]
    per_source_rows: list[dict[str, Any]] = []
    for gen in sorted(df_ok["generator"].astype(str).unique()):
        gm = (df_ok["generator"].astype(str) == gen).to_numpy()
        cls_is_fake = int(df_ok.loc[gm, "label"].iloc[0]) == fake_idx
        role = str(df_ok.loc[gm, "split_role"].iloc[0])
        abstain_g = float((labels[gm] == "UNCERTAIN").mean())
        dec_g = gm & (labels != "UNCERTAIN")
        bin_rate = float((bin_labels[gm] == "FAKE").mean())
        dec_rate = float((labels[dec_g] == "FAKE").mean()) if dec_g.any() else 0.0
        auc_g = (
            float(
                roc_auc_score(
                    np.concatenate([np.zeros(len(real_scores_all)), np.ones(gm.sum())]),
                    np.concatenate([real_scores_all, p_cal[gm]]),
                )
            )
            if cls_is_fake
            else None
        )
        per_source_rows.append(
            {
                "source": gen,
                "class": "FAKE" if cls_is_fake else "REAL",
                "role": role,
                "n": int(gm.sum()),
                "metric": "Recall" if cls_is_fake else "FPR",
                "binary_rate": bin_rate,
                "decided_rate": dec_rate,
                "uncertain_rate": abstain_g,
                "view_disagreements": int(vd[gm].sum()),
                "branch_disagreements": int(bd[gm].sum()),
                "auroc_vs_test_real": auc_g,
            }
        )

    test_pert_parts = [
        grp.sample(n=min(args.pert_per_source, len(grp)), random_state=args.seed + 11)
        for _, grp in test_df.groupby("generator")
    ]
    test_pert_df = pd.concat(test_pert_parts, ignore_index=True)
    pert_rows: list[dict[str, Any]] = []
    for p_name in STEP3_EVAL_PERTS:
        tp_raw = _collect_raw_branch_scalars(
            test_pert_df, clip_head, conv_model, conv_tf, te_cache, sha_to_te,
            pert_name=p_name, fake_idx=fake_idx, real_idx=real_idx,
            batch_size=args.batch_size, device=device, require_face=True,
        )
        tp = _blend_scalars(tp_raw, w_clip)
        pm = _evaluate_decisions(
            tp,
            temperature=temperature,
            bias=bias,
            decision_threshold=decision_thr,
            uncertain_band=(band[0], band[1]),
            view_margin=args.view_margin,
            branch_margin=args.branch_margin,
        )
        pert_rows.append(
            {
                "perturbation": p_name,
                "n": pm["n"],
                "auroc": pm["auroc"],
                "ece_before": pm["ece_before"],
                "ece_after": pm["ece_after"],
                "binary_recall": pm["binary_recall"],
                "binary_real_fpr": pm["binary_real_fpr"],
                "decided_recall": pm["decided_recall"],
                "decided_real_fpr": pm["decided_real_fpr"],
                "abstention_rate": pm["abstention_rate"],
                "view_disagreements": pm["n_view_disagreement"],
                "branch_disagreements": pm["n_branch_disagreement"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_clean = {k: v for k, v in test_overall.items() if not isinstance(v, np.ndarray)}
    save_json(
        {
            "calibration": cal_payload,
            "held_out_overall": summary_clean,
            "held_out_per_source": per_source_rows,
            "held_out_perturbations": pert_rows,
        },
        args.out_dir / "calibration_summary.json",
    )

    md = [
        "# Step 4 Calibrated Ensemble & Decision Logic Report\n",
        f"- Ensemble: `CLIP-L/14 ({best_clip}, w={w_clip:.2f}) + convnext_v2 (w={1.0 - w_clip:.2f})` (`fit_ood_weight={args.fit_ood_weight}`)",
        f"- Bounded calibration (fitted strictly on `val` clean + perturbed): **`T = {temperature:.4f}`** (`T >= 1.0` enforced), shift **`b = {bias:+.4f}`**",
        f"- Validation NLL: `{cal_fit['val_nll_before']:.4f} -> {cal_fit['val_nll_after']:.4f}` | Validation ECE: `{cal_fit['val_ece_before']:.4f} -> {cal_fit['val_ece_after']:.4f}`",
        f"- Decision threshold (clean `val_real` target FPR <= `{100*args.target_val_fpr:.1f}%`): **`{decision_thr:.4f}`** (empirical `val_real` FPR `{100*actual_val_fpr:.2f}%`)",
        f"- Uncertain band (+/- `{args.band_half_logit:.2f}` log-odds): **`[{band[0]:.4f}, {band[1]:.4f}]`** | View margin: `{args.view_margin:.2f}` | Branch margin: `{args.branch_margin:.2f}`\n",
        "## 1. Overall Held-Out Test Calibration & Selective Metrics (`5,435` face-detected images)\n",
        "| Metric | Value |",
        "|---|---:|",
        f"| AUROC | {summary_clean['auroc']:.4f} |",
        f"| Test NLL (before -> after) | {summary_clean['nll_before']:.4f} -> {summary_clean['nll_after']:.4f} |",
        f"| Test ECE (before -> after) | {summary_clean['ece_before']:.4f} -> {summary_clean['ece_after']:.4f} |",
        f"| Binary Recall (all samples @ {decision_thr:.4f}) | {100*summary_clean['binary_recall']:.1f}% |",
        f"| Binary Real FPR (all samples @ {decision_thr:.4f}) | {100*summary_clean['binary_real_fpr']:.1f}% |",
        f"| Decided Recall (non-UNCERTAIN) | {100*summary_clean['decided_recall']:.1f}% |",
        f"| Decided Real FPR (non-UNCERTAIN) | {100*summary_clean['decided_real_fpr']:.1f}% |",
        f"| UNCERTAIN Abstention Rate | {100*summary_clean['abstention_rate']:.1f}% ({summary_clean['n_uncertain']}/{summary_clean['n']}) |",
        f"| Abstentions breakdown (view_disagreement / branch_disagreement / borderline) | {summary_clean['n_view_disagreement']} / {summary_clean['n_branch_disagreement']} / {summary_clean['n_borderline']} |",
        f"| Binary error rate on abstained samples | {100*summary_clean['abstained_binary_error_rate']:.1f}% |\n",
        "## 2. Per-Source Breakdown on Held-Out Test\n",
        "| Source | Class | Role | n | Metric | Binary Rate | Decided Rate | UNCERTAIN Rate | View/Branch Disagreements | AUROC vs Test Real |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for s in per_source_rows:
        auc_s = "-" if s["auroc_vs_test_real"] is None else f"{s['auroc_vs_test_real']:.4f}"
        md.append(
            f"| `{s['source']}` | {s['class']} | `{s['role']}` | {s['n']} | {s['metric']} | "
            f"{100*s['binary_rate']:.1f}% | {100*s['decided_rate']:.1f}% | {100*s['uncertain_rate']:.1f}% | "
            f"{s['view_disagreements']} / {s['branch_disagreements']} | {auc_s} |"
        )
    md.append("\n## 3. Perturbation Robustness (`held_out_test`, `30`/source)\n")
    md.append("| Perturbation | n | AUROC | ECE (before -> after) | Binary Rec / FPR | Decided Rec / FPR | UNCERTAIN % | View/Branch Disagreements |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for p in pert_rows:
        md.append(
            f"| `{p['perturbation']}` | {p['n']} | {p['auroc']:.4f} | "
            f"{p['ece_before']:.4f} -> {p['ece_after']:.4f} | "
            f"{100*p['binary_recall']:.1f}% / {100*p['binary_real_fpr']:.1f}% | "
            f"{100*p['decided_recall']:.1f}% / {100*p['decided_real_fpr']:.1f}% | "
            f"{100*p['abstention_rate']:.1f}% | {p['view_disagreements']} / {p['branch_disagreements']} |"
        )
    report_path = args.out_dir / "calibration_report.md"
    report_path.write_text("\n".join(md) + "\n")
    logger.info("Saved Step 4 report -> %s", report_path)


if __name__ == "__main__":
    main()
