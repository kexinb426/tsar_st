import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping {path} line {i}: {e}")
    return rows

def _dedupe_keep_last(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = {}
    for r in rows:
        k = r.get(key)
        if k is not None:
            seen[k] = r  # keep last
    return list(seen.values())

def aggregate_pairs(
    results_dir: str = "results",
    save_dir: str = "a_misc./agg_results",
    join_type: str = "inner",           # "inner" ensures both text + scores exist
    flatten_probas: bool = False,        # flatten cefr_probas into wide columns
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    combined_frames = []

    print(f"🔍 Scanning '{results_dir}'...")
    for run in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, run)
        if not os.path.isdir(run_dir):
            continue

        simp_path  = os.path.join(run_dir, "simplifications.jsonl")
        score_path = os.path.join(run_dir, "individual_scores.jsonl")
        if not (os.path.exists(simp_path) and os.path.exists(score_path)):
            continue

        # Read & dedupe
        simp_rows  = _dedupe_keep_last(_read_jsonl(simp_path),  "text_id")
        score_rows = _dedupe_keep_last(_read_jsonl(score_path), "text_id")

        df_s = pd.DataFrame(simp_rows)
        df_q = pd.DataFrame(score_rows)

        if df_s.empty or df_q.empty:
            continue

        # Optionally flatten nested cefr_probas
        if flatten_probas and "cefr_probas" in df_q.columns:
            flat = pd.json_normalize(
                df_q["cefr_probas"].apply(lambda d: d if isinstance(d, dict) else {})
            )
            flat.columns = [f"cefr_probas_{c.replace('.', '_')}" for c in flat.columns]
            df_q = pd.concat([df_q.drop(columns=["cefr_probas"]), flat], axis=1)

        # Core columns to keep
        keep_simpl_cols = [
            "text_id", "simplified", "raw_model_output", "few_shot_example_ids"
        ]
        keep_score_cols = [
            "text_id", "target_cefr", "predicted_cefr", "predicted_cefr_confidence",
            "cefr_adj_accuracy", "meaningbert_orig", "bertscore_f1_orig",
            "meaningbert_ref", "bertscore_f1_ref"
        ]
        df_s = df_s[[c for c in keep_simpl_cols if c in df_s.columns]].copy()
        df_q = df_q[[c for c in keep_score_cols if c in df_q.columns] +
                    [c for c in df_q.columns if c.startswith("cefr_probas_")]].copy()

        # Merge on text_id
        merged = pd.merge(df_s, df_q, on="text_id", how=join_type)
        if merged.empty:
            continue

        merged["experiment_run"] = run
        combined_frames.append(merged)

    if not combined_frames:
        print("❌ No mergeable data found.")
        return

    all_rows = pd.concat(combined_frames, ignore_index=True)

    # Nice column order
    front = ["experiment_run", "text_id", "simplified", "raw_model_output"]
    metrics = [
        "target_cefr", "predicted_cefr", "predicted_cefr_confidence",
        "cefr_adj_accuracy", "meaningbert_orig", "bertscore_f1_orig",
        "meaningbert_ref", "bertscore_f1_ref"
    ]
    probas = [c for c in all_rows.columns if c.startswith("cefr_probas_")]
    rest = [c for c in all_rows.columns if c not in set(front + metrics + probas)]
    ordered_cols = [c for c in front if c in all_rows.columns] + \
                   [c for c in metrics if c in all_rows.columns] + \
                   probas + rest
    all_rows = all_rows[ordered_cols]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(save_dir, f"simplifications_with_scores_{ts}.csv")
    all_rows.to_csv(out_csv, index=False)
    print(f"✅ Wrote: {out_csv}")

if __name__ == "__main__":
    aggregate_pairs()