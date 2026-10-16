"""Recovered preprocessing from the supplied hierarchical-model notebook.

Original text cleaning, deduplication and split algorithms are preserved.
Input plumbing and aggregate reporting are new. Outputs contain local dataset
records and must remain excluded from a public repository.
"""
import argparse
import ast
import html
import json
import os
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")

def normalize_unicode(s: str) -> str:
    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)
    return s

def fix_escaped_unicode(s: str) -> str:
    def _unescape(m):
        code = m.group(0) 
        try:
            return chr(int(code[2:], 16))
        except Exception:
            return code
    return re.sub(r"\\u[0-9a-fA-F]{4}", _unescape, s)

def strip_markdown_links_keep_text(s: str) -> str:
    return MD_LINK_RE.sub(r"\1", s)

def replace_urls_with_token(s: str) -> str:
    return URL_RE.sub(" [URL] ", s)

def normalize_whitespace_keep_boundaries(s: str) -> str:
    s = s.replace("\u200b", "") 
    s = re.sub(r"\s*###\s*", ". ", s)                     
    s = re.sub(r"(?:(?:-|\*){3,}|/{2,}|\\{2,})", ". ", s) 
    s = re.sub(r"\n{3,}", "\n\n", s)

    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)

    return s.strip()

def clean_text_core(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    s = fix_escaped_unicode(s)
    s = normalize_unicode(s)
    s = strip_markdown_links_keep_text(s)
    s = replace_urls_with_token(s)
    s = normalize_whitespace_keep_boundaries(s)

    return s

def _norm_field(x):
    x = (x or "").strip()
    return "" if x.lower() == "nan" else x

def build_text(title: str, post: str, text: str) -> str:
    title = _norm_field(title)
    post  = _norm_field(post)
    text  = _norm_field(text)

    if title and post and text.startswith(title) and "###" in text:
        base = f"{title}. {post}"
    elif post:
        base = f"{title + '. ' if title else ''}{post}"
    elif title:
        base = title
    else:
        base = text

    return clean_text_core(base)


def prepare(data_directory, output_directory):
    import pandas as pd
    import numpy as np
    df = pd.concat([pd.read_json(Path(data_directory) / name, lines=True)
                        for name in ("train.json", "val.json", "test.json")], ignore_index=True)
    raw_count = len(df)
    for col in ["title", "post", "text"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")  
                .astype(str)
                .str.replace(r"[ \t]+", " ", regex=True) 
                .str.strip()
            )

    df["text_clean"] = df.apply(
        lambda r: build_text(r.get("title"), r.get("post"), r.get("text")),
        axis=1
    )


    before = len(df)

    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    df = df[df["text_clean"].str.strip().ne("")]

    removed_empty = before - len(df)

    removed_id = 0
    if "id" in df.columns:
        b = len(df)
        df = df.drop_duplicates(subset=["id"])
        removed_id = b - len(df)
    
    if "label_id" in df.columns and "label_id_bits" not in df.columns:
        df["label_id_bits"] = df["label_id"].astype(str).str.zfill(8)

    b = len(df)
    dedup_cols = ["text_clean"]
    if "label_id_bits" in df.columns:
        dedup_cols.append("label_id_bits")

    df = df.drop_duplicates(subset=dedup_cols)

    removed_textdup = b - len(df)

    print(
        f"Removed empty: {removed_empty} | "
        f"Removed dup id: {removed_id} | "
        f"Removed dup text_clean: {removed_textdup} | "
        f"Now: {len(df)}"
    )
    EXPECTED = {
        "anger",
        "brain dysfunction (forget)",
        "emptiness",
        "hopelessness",
        "loneliness",
        "sadness",
        "suicide intent",
        "worthlessness",
    }

    def to_list_clean(x):
        """
        Convert various formats to a clean list[str]:
        - Handles list, stringified list, single string
        - Removes NaN/None/empty
        - Normalizes to lowercase
        - Deduplicates while preserving order
        """
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []

        if isinstance(x, list):
            items = x
        else:
            s = str(x).strip()
            if not s or s.lower() == "nan":
                return []
  
            try:
                v = ast.literal_eval(s)
                items = v if isinstance(v, list) else [s]
            except Exception:
                items = [s]


        norm = []
        seen = set()
        for it in items:
            t = str(it).strip().lower()
            if not t or t == "nan":
                continue
            if t not in seen:
                norm.append(t)
                seen.add(t)

        return norm

    if "emotions" in df.columns:

        df["emotions"] = df["emotions"].apply(to_list_clean)


        seen_all = set(e for L in df["emotions"] for e in L)
        unknown = sorted(seen_all - EXPECTED)
        print("Unknown labels (data - expected):", unknown)

        df["emotions"] = df["emotions"].apply(lambda L: [e for e in L if e in EXPECTED])

        before = len(df)

    if "label_id_bits" not in df.columns:
        df = df[df["emotions"].apply(len) > 0]

        print("Dropped rows with empty labels after filtering:", before - len(df))

    LABELS = [
        "anger","brain dysfunction (forget)","emptiness","hopelessness",
        "loneliness","sadness","suicide intent","worthlessness"
    ]

    def emotions_to_bitstr(L):
        S = set(L)
        return "".join("1" if lab in S else "0" for lab in LABELS)

    if "label_id" in df.columns and "label_id_bits" not in df.columns:
        df["label_id_bits"] = df["label_id"].astype(str).str.zfill(8)

    if "emotions" in df.columns and "label_id_bits" in df.columns:
        df["mask_bits_from_emotions"] = df["emotions"].apply(emotions_to_bitstr)
        mismatch = df["mask_bits_from_emotions"] != df["label_id_bits"]
        print(f"label_id mismatch: {mismatch.sum()} rows | rate={mismatch.mean():.4f}")

    assert "label_id_bits" in df.columns, "label_id_bits missing — cannot build y"
    df[
    "y"] = df["label_id_bits"].apply(lambda s: [int(c) for c in
     s])
    Y = np.vstack(df[
    "y"].to_numpy()).astype(int
    )
    print("Y shape:"
    , Y.shape)
    import os
    import json
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # -----------------------------
    # Split settings (reproducible)
    # -----------------------------
    RANDOM_STATE = 123
    OUT_DIR = str(output_directory)  # keep consistent with cfg.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------------
    # Basic checks
    # -----------------------------
    assert "text_clean" in df.columns, "Missing column: text_clean"
    assert "y" in df.columns, "Missing column: y (multi-hot list/array)"

    # Canonical label order (must match model output order)
    LABELS = [
        "anger",
        "brain dysfunction (forget)",
        "emptiness",
        "hopelessness",
        "loneliness",
        "sadness",
        "suicide intent",
        "worthlessness",
    ]

    # -----------------------------
    # Build X/Y
    # -----------------------------
    X = df["text_clean"].astype(str).to_numpy()

    # Ensure y is a 2D numpy array (N, L)
    Y = np.vstack(df["y"].to_numpy()).astype(int)
    assert Y.ndim == 2, "Y must be 2D (N, L)"
    assert Y.shape[1] == len(LABELS), f"Y has {Y.shape[1]} labels, but LABELS has {len(LABELS)}"

    # -----------------------------
    # A lightweight stratification key for multi-label data
    # (to avoid extra dependencies)
    #   - Keep balance on: (suicide intent present?) + (number of positive labels)
    # -----------------------------
    suicide_idx = LABELS.index("suicide intent")
    pos_count = Y.sum(axis=1)
    pos_count_bucket = np.clip(pos_count, 0, 3)  # 0,1,2,3(=3+)
    strat_key = (Y[:, suicide_idx] * 10 + pos_count_bucket).astype(int)

    # -----------------------------
    # Train/Val/Test split
    # -----------------------------
    try:
        idx_all = np.arange(len(X))
        idx_trainval, idx_test = train_test_split(
            idx_all,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=strat_key
        )

        strat_key_trainval = strat_key[idx_trainval]
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=0.20,  # 0.2 of 0.8 -> 0.16 overall
            random_state=RANDOM_STATE,
            stratify=strat_key_trainval
        )
        print("Split created with stratification key.")
    except Exception as e:
        print("WARNING: Stratified split failed. Falling back to random split.")
        print("Reason:", repr(e))

        idx_all = np.arange(len(X))
        idx_trainval, idx_test = train_test_split(
            idx_all,
            test_size=0.20,
            random_state=RANDOM_STATE,
            shuffle=True
        )
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=0.20,
            random_state=RANDOM_STATE,
            shuffle=True
        )

    # -----------------------------
    # Build split DataFrames
    # -----------------------------
    def _subset_df(indices: np.ndarray) -> pd.DataFrame:
        sub = df.iloc[indices].copy()
        sub = sub[["text_clean", "y"]]
        # Convert y to plain python lists for JSONL safety
        sub["y"] = sub["y"].apply(lambda v: list(v) if not isinstance(v, list) else v)
        return sub.reset_index(drop=True)

    train_df = _subset_df(idx_train)
    val_df = _subset_df(idx_val)
    test_df = _subset_df(idx_test)

    # -----------------------------
    # Save JSONL + labels.json
    # -----------------------------
    train_path = os.path.join(OUT_DIR, "train.jsonl")
    val_path = os.path.join(OUT_DIR, "val.jsonl")
    test_path = os.path.join(OUT_DIR, "test.jsonl")
    labels_path = os.path.join(OUT_DIR, "labels.json")

    train_df.to_json(train_path, orient="records", lines=True, force_ascii=False)
    val_df.to_json(val_path, orient="records", lines=True, force_ascii=False)
    test_df.to_json(test_path, orient="records", lines=True, force_ascii=False)

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=2)

    print("Saved split files:")
    print("  Train:", train_path, "N=", len(train_df))
    print("  Val:  ", val_path,   "N=", len(val_df))
    print("  Test: ", test_path,  "N=", len(test_df))
    print("Saved labels:", labels_path)

    # -----------------------------
    # Quick sanity stats
    # -----------------------------
    def _label_prevalence(y_list, labels):
        yy = np.vstack(y_list).astype(int)
        return pd.Series(yy.mean(axis=0), index=labels).sort_values(ascending=False)

    print("\nLabel prevalence (train):")
    print(_label_prevalence(train_df["y"].tolist(), LABELS).round(4))

    print("\nLabel prevalence (val):")
    print(_label_prevalence(val_df["y"].tolist(), LABELS).round(4))

    print("\nLabel prevalence (test):")
    print(_label_prevalence(test_df["y"].tolist(), LABELS).round(4))

    split_frames = {"train": train_df, "validation": val_df, "test": test_df}
    text_sets = {k:set(v["text_clean"]) for k,v in split_frames.items()}
    summary = {
        "status": "preprocessing_and_split_executed_locally_no_model_training",
        "input_order": ["train.json", "val.json", "test.json"],
        "raw_records": raw_count,
        "cleaned_records": len(df),
        "removed_empty_text": removed_empty,
        "removed_duplicate_ids": removed_id,
        "removed_duplicate_cleaned_text_and_label": removed_textdup,
        "remaining_duplicate_cleaned_text_excess_records": int(df["text_clean"].duplicated().sum()),
        "seed": RANDOM_STATE,
        "stratification": "suicide_intent_bit * 10 + min(positive_label_count, 3)",
        "splits": {k:{"records":len(v), "positive_label_assignments":int(np.vstack(v["y"]).sum()),
                       "per_label_counts":dict(zip(LABELS,np.vstack(v["y"]).sum(axis=0).astype(int).tolist()))}
                   for k,v in split_frames.items()},
        "shared_cleaned_texts_between_splits": {a+"_vs_"+b:len(text_sets[a]&text_sets[b])
               for a,b in [("train","validation"),("train","test"),("validation","test")]},
        "runtime_versions": {"numpy": np.__version__, "pandas":pd.__version__},
        "limits": ["Matches saved notebook split counts and label prevalence; original split IDs are not available.",
                   "Model weights and prediction arrays have not been supplied or regenerated.",
                   "Deduplication retains identical cleaned texts with different label vectors."]
    }
    with open(Path(output_directory)/"preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory",type=Path)
    parser.add_argument("--output",type=Path,default=Path("outputs_hier_strategy4"))
    args=parser.parse_args()
    print(json.dumps(prepare(args.data_directory,args.output),indent=2))
