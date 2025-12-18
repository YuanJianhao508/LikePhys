#!/usr/bin/env python3
import os
import json
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = "./results/evaluation_t10_uniform_1_cfg_final"
# ────────────────────────────────────────────────────────────────────────────────

# Model name mapping for display (keys must match JSON filenames)
name_map = {
    "animatediff":      "AnimateDiff",
    "animatediff_sdxl": "AnimateDiff SDXL",
    "modelscope":       "ModelScope",
    "zeroscope":        "ZeroScope",
    "cogvideox":        "CogVideoX",
    "cogvideox-5b":     "CogVideoX–5B",
    "cogvideox1.5-5b":  "CogVideoX1.5–5B",
    "hunyuan_t2v":      "Hunyuan T2V",
    "wan2.1-T2V-1.3b":  "Wan2.1-T2V-1.3B",
    "wan2.1-T2V-14b":   "Wan2.1-T2V-14B",
    "ltx":              "LTX",
    "ltx-0.9.1":        "LTX v0.9.1",
    "ltx-0.9.5":        "LTX v0.9.5",
    "mochi":            "Mochi",
}

# Per-dataset variation filters (if any)
filter_config = {
    "shadowm":  ["shadow_disconnection"],
    "pendulum": ["reverse_gravity", "break_conservation_energy"],
    "river":    ["color_change", "teleporting_fluid", "negative_viscosity"],
}

ignore_models = ["svd"]

# Desired display order of models
model_display_order = [
    "animatediff", "animatediff_sdxl", "modelscope", "zeroscope",
    "cogvideox", "cogvideox-5b", "cogvideox1.5-5b", "hunyuan_t2v",
    "wan2.1-T2V-1.3b", "wan2.1-T2V-14b", "ltx", "ltx-0.9.1", "ltx-0.9.5", "mochi"
]

# ─── RANKING METHOD CONFIGURATION ─────────────────────────────────────────────
# Choose ranking method:
# - "dataset_weighted": Each dataset has equal weight (current default)
# - "variation_weighted": Each variation has equal weight (matches read_exp_v2.py)
RANKING_METHOD = "variation_weighted"  # Change this to "dataset_weighted" for old behavior
# ──────────────────────────────────────────────────────────────────────────────

def compute_misrank_normalized(results, dataset_name):
    invalid_types = sorted({
        v for sub in results.values() for v in sub.keys() if v != "valid"
    })
    out = {}
    for var in invalid_types:
        if dataset_name in filter_config and var in filter_config[dataset_name]:
            continue
        ratios = []
        for sub in results.values():
            if "valid" not in sub or var not in sub:
                continue
            valid_losses = [np.mean(info["loss_array"]) for info in sub["valid"].values()]
            invalid_losses = [np.mean(info["loss_array"]) for info in sub[var].values()]
            pairs = [(v, i) for v in valid_losses for i in invalid_losses]
            if not pairs:
                continue
            mis = sum(1 for v, i in pairs if v > i)
            ratios.append(mis / len(pairs))
        if not ratios:
            raise ValueError(f"No valid ratios found for dataset '{dataset_name}', variation '{var}'")
        out[var] = float(np.mean(ratios))
    return out

def main():
    # 1) Load and compute per-variation mis-rank
    summary = {}
    for ds in sorted(os.listdir(BASE_DIR)):
        ds_path = os.path.join(BASE_DIR, ds)
        if not os.path.isdir(ds_path):
            continue
        summary[ds] = {}
        for fn in sorted(os.listdir(ds_path)):
            if not fn.startswith("results_") or not fn.endswith(".json"):
                continue
            model = fn[len("results_"):-len(".json")]
            if model in ignore_models:
                continue
            data = json.load(open(os.path.join(ds_path, fn)))
            summary[ds][model] = compute_misrank_normalized(
                data.get("scene_evaluations", {}), ds
            )

    # 2) Compute dataset-level averages
    dataset_model_avg = {}
    for ds, mdl_dict in summary.items():
        dataset_model_avg[ds] = {
            m: float(np.mean(list(vars_.values()))) if vars_ else np.nan
            for m, vars_ in mdl_dict.items()
        }

    # 3) Determine which models appear
    all_models = [m for m in model_display_order if any(m in summary[ds] for ds in summary)]

    # 4) Compute overall mean mis-rank per model
    all_datasets = sorted(dataset_model_avg.keys())
    
    if RANKING_METHOD == "dataset_weighted":
        # Original method: Each dataset has equal weight
        model_overall = {
            m: np.nanmean([dataset_model_avg[ds].get(m, np.nan) for ds in all_datasets])
            for m in all_models
        }
        print(f"Using dataset-weighted averaging (each dataset has equal weight)")
    elif RANKING_METHOD == "variation_weighted":
        # New method: Each variation has equal weight (matches read_exp_v2.py)
        model_scores = {m: [] for m in all_models}
        for ds in summary.values():
            for m, var_dict in ds.items():
                if m in all_models:  # Only include models we're tracking
                    model_scores[m].extend(var_dict.values())
        model_overall = {m: float(np.mean(scores)) if scores else np.nan 
                        for m, scores in model_scores.items()}
        print(f"Using variation-weighted averaging (each variation has equal weight)")
    else:
        raise ValueError(f"Invalid RANKING_METHOD: {RANKING_METHOD}. Use 'dataset_weighted' or 'variation_weighted'")

    # 5) Sort models by overall performance
    if RANKING_METHOD == "dataset_weighted":
        # Original behavior: worst→best so best at bottom (for LaTeX table)
        sorted_models = sorted(all_models, key=lambda m: model_overall[m], reverse=True)
        print(f"Table sorted worst→best (best models at bottom)")
    elif RANKING_METHOD == "variation_weighted":
        # Match read_exp_v2.py behavior: best→worst (for consistency)
        sorted_models = sorted(all_models, key=lambda m: model_overall[m])
        print(f"Table sorted best→worst (best models at top)")
    else:
        # This shouldn't happen due to earlier check, but just in case
        sorted_models = sorted(all_models, key=lambda m: model_overall[m], reverse=True)

    # 6) Precompute per-column rankings
    col_rankings = {}
    for ds in all_datasets:
        vals = [
            (m, dataset_model_avg[ds][m])
            for m in sorted_models
            if not np.isnan(dataset_model_avg[ds].get(m, np.nan))
        ]
        col_rankings[ds] = sorted(vals, key=lambda x: x[1])

    # 7) Print LaTeX table of dataset-level averages
    print(r"\begin{table}[ht]")
    print(r"  \centering")
    print(r"  \caption{Average mis-rank (\%) by model and dataset (lower is better).}")
    print(r"  \resizebox{\linewidth}{!}{\begin{tabular}{l" + "c"*len(all_datasets) + "}")
    print(r"    \toprule")
    header = "    Model & " + " & ".join(ds.replace("_", " ").title() for ds in all_datasets) + r" \\"
    print(header)
    print(r"    \midrule")

    for m in sorted_models:
        disp = name_map.get(m, m.replace("_", " ").title())
        cells = []
        for ds in all_datasets:
            v = dataset_model_avg[ds].get(m, np.nan)
            if np.isnan(v):
                cells.append("")
            else:
                pct = v * 100
                best = col_rankings[ds][0][0]
                second = col_rankings[ds][1][0] if len(col_rankings[ds]) > 1 else None
                text = f"{pct:.1f}"
                if m == best:
                    cells.append(rf"\textbf{{{text}}}")
                elif m == second:
                    cells.append(rf"\underline{{{text}}}")
                else:
                    cells.append(text)
        print("    " + disp + " & " + " & ".join(cells) + r" \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}}")
    print(r"  \label{tab:dataset_avgs}")
    print(r"\end{table}")
    
    # 8) Print overall ranking for verification (similar to read_exp_v2.py)
    print(f"\n{'='*60}")
    print(f"OVERALL MODEL RANKING ({RANKING_METHOD.upper()})")
    print(f"{'='*60}")
    
    if RANKING_METHOD == "variation_weighted":
        # Sort best to worst for display (like read_exp_v2.py)
        ranking_display = sorted(model_overall.items(), key=lambda kv: kv[1])
        print("Model ranking by average mis-rank rate across all variations (lower is better):\n")
        for i, (m, avg) in enumerate(ranking_display, 1):
            disp = name_map.get(m, m.replace("_", " ").title())
            print(f"{i}. {disp}: {avg:.3f}")
    else:
        # Sort worst to best for display
        ranking_display = sorted(model_overall.items(), key=lambda kv: kv[1], reverse=True)
        print("Model ranking by average mis-rank rate across datasets (higher is worse):\n")
        for i, (m, avg) in enumerate(ranking_display, 1):
            disp = name_map.get(m, m.replace("_", " ").title())
            print(f"{i}. {disp}: {avg:.3f} (worst)")

if __name__ == "__main__":
    main()
