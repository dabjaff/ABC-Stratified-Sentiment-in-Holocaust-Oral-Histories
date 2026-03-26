from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
except ImportError as e:
    raise ImportError("This script requires scikit-learn. Install with: pip install scikit-learn") from e


SENTENCE_CSV = "?????/sentence_level_abc_details.csv"
UTTERANCE_CSV = "????3/utterance_level_abc.csv"
OUTPUT_FOLDER_NAME = "kappa_agreement_outputs"

SENTENCE_COLS = {
    "SiE": "SBert_Sentiment",
    "Car": "Card_Sentiment",
    "NLT": "NLPTown_Sentiment",
}

UTTERANCE_COLS = {
    "SiE": "SBert_AggregatedSentiment",
    "Car": "Card_AggregatedSentiment",
    "NLT": "NLPTown_AggregatedSentiment",
}

PRETTY = {
    "SiE": r"\textsc{SiEBERT}",
    "Car": r"\textsc{Cardiff}",
    "NLT": r"\textsc{NLPTown}",
}

LABELS_3 = [-1, 0, 1]
LABELS_2 = [-1, 1]


def to_ternary(x) -> float:
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, np.integer)):
        if x in (-1, 0, 1):
            return float(x)
        if x in (1, 2):
            return -1.0
        if x == 3:
            return 0.0
        if x in (4, 5):
            return 1.0
        return np.nan

    s = str(x).strip().lower()

    if s in {"1", "2"}:
        return -1.0
    if s == "3":
        return 0.0
    if s in {"4", "5"}:
        return 1.0

    if s in {"positive", "pos", "+1", "1"}:
        return 1.0
    if s in {"negative", "neg", "-1"}:
        return -1.0
    if s in {"neutral", "neu", "0"}:
        return 0.0

    if "pos" in s or "positive" in s:
        return 1.0
    if "neg" in s or "negative" in s:
        return -1.0
    if "neu" in s or "neutral" in s:
        return 0.0

    return np.nan


def fleiss_kappa(counts: np.ndarray) -> float:
    M = np.asarray(counts, dtype=float)
    N, k = M.shape
    n = M.sum(axis=1)
    if not np.allclose(n, n[0]):
        raise ValueError("Fleiss requires the same number of ratings per item.")
    n = float(n[0])

    P_i = (M * (M - 1)).sum(axis=1) / (n * (n - 1))
    P_bar = P_i.mean()

    p_j = M.sum(axis=0) / (N * n)
    P_e = (p_j ** 2).sum()

    denom = 1.0 - P_e
    if np.isclose(denom, 0.0):
        return np.nan
    return (P_bar - P_e) / denom


def pairwise_metrics(df: pd.DataFrame, a: str, b: str) -> Dict[str, float]:
    x = df[a].astype(int).to_numpy()
    y = df[b].astype(int).to_numpy()

    N = len(df)
    agr = 100.0 * float((x == y).mean())
    kap = float(cohen_kappa_score(x, y, labels=LABELS_3))

    mask = (df[a] != 0) & (df[b] != 0)
    sub = df.loc[mask, [a, b]].copy()
    N_neg0 = len(sub)

    if N_neg0 == 0:
        agr_neg0 = np.nan
        kap_neg0 = np.nan
    else:
        xs = sub[a].astype(int).to_numpy()
        ys = sub[b].astype(int).to_numpy()
        agr_neg0 = 100.0 * float((xs == ys).mean())
        kap_neg0 = float(cohen_kappa_score(xs, ys, labels=LABELS_2))

    return {
        "N": N,
        "Agr": agr,
        "kappa": kap,
        "N_neg0": N_neg0,
        "Agr_neg0": agr_neg0,
        "kappa_neg0": kap_neg0,
    }


def confusion_outputs(
    df: pd.DataFrame,
    a: str,
    b: str,
    out_csv_raw: Path,
    out_csv_norm: Path,
    labels: List[int],
) -> None:
    x = df[a].astype(int).to_numpy()
    y = df[b].astype(int).to_numpy()
    cm = confusion_matrix(x, y, labels=labels)

    raw_df = pd.DataFrame(cm, index=[f"{a}_{v}" for v in labels], columns=[f"{b}_{v}" for v in labels])
    raw_df.to_csv(out_csv_raw, index=True)

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cmn = np.divide(cm, row_sums, where=row_sums != 0)
    norm_df = pd.DataFrame(cmn, index=raw_df.index, columns=raw_df.columns)
    norm_df.to_csv(out_csv_norm, index=True)


def compute_fleiss(df: pd.DataFrame, cols: List[str], labels: List[int]) -> float:
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    counts = np.zeros((len(df), len(labels)), dtype=int)

    vals = df[cols].astype(int).to_numpy()
    for i in range(len(df)):
        for r in range(vals.shape[1]):
            counts[i, label_to_idx[vals[i, r]]] += 1

    return float(fleiss_kappa(counts))


def load_and_prepare(csv_path: str, colmap: Dict[str, str]) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    missing = [src for src in colmap.values() if src not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    for short, src in colmap.items():
        out[short] = df[src].apply(to_ternary)

    out = out.dropna(subset=list(colmap.keys())).copy()
    out[list(colmap.keys())] = out[list(colmap.keys())].astype(int)
    return out


def main():
    sent_path = Path(SENTENCE_CSV)
    out_dir = sent_path.parent / OUTPUT_FOLDER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    df_sent = load_and_prepare(SENTENCE_CSV, SENTENCE_COLS)
    df_utt = load_and_prepare(UTTERANCE_CSV, UTTERANCE_COLS)

    pairs = [("SiE", "Car"), ("SiE", "NLT"), ("Car", "NLT")]
    rows = []

    def process_level(df_level: pd.DataFrame, level_name: str):
        nonlocal rows

        for a, b in pairs:
            m = pairwise_metrics(df_level, a, b)
            cm_raw = out_dir / f"confusion_{level_name}_{a}_vs_{b}_raw.csv"
            cm_norm = out_dir / f"confusion_{level_name}_{a}_vs_{b}_row_normalized.csv"
            confusion_outputs(df_level, a, b, cm_raw, cm_norm, LABELS_3)

            rows.append({
                "Level": level_name,
                "Pair": f"{a}--{b}",
                "N": m["N"],
                "Agr": m["Agr"],
                "kappa": m["kappa"],
                "N_neg0": m["N_neg0"],
                "Agr_neg0": m["Agr_neg0"],
                "kappa_neg0": m["kappa_neg0"],
            })

        fleiss_3 = compute_fleiss(df_level, ["SiE", "Car", "NLT"], LABELS_3)
        df_pol = df_level[(df_level["Car"] != 0) & (df_level["NLT"] != 0)].copy()
        fleiss_2 = compute_fleiss(df_pol, ["SiE", "Car", "NLT"], LABELS_2)

        return fleiss_3, fleiss_2, len(df_level), len(df_pol)

    fleiss_sent_3, fleiss_sent_2, N_sent, N_sent_pol = process_level(df_sent, "Sentence")
    fleiss_utt_3, fleiss_utt_2, N_utt, N_utt_pol = process_level(df_utt, "Utterance")

    summary = pd.DataFrame(rows)
    summary_fmt = summary.copy()
    summary_fmt["Agr"] = summary_fmt["Agr"].map(lambda x: round(float(x), 1))
    summary_fmt["kappa"] = summary_fmt["kappa"].map(lambda x: round(float(x), 3))
    summary_fmt["Agr_neg0"] = summary_fmt["Agr_neg0"].map(
        lambda x: round(float(x), 1) if pd.notna(x) else np.nan
    )
    summary_fmt["kappa_neg0"] = summary_fmt["kappa_neg0"].map(
        lambda x: round(float(x), 3) if pd.notna(x) else np.nan
    )

    out_summary_csv = out_dir / "pairwise_kappa_summary.csv"
    summary_fmt.to_csv(out_summary_csv, index=False)

    out_fleiss = out_dir / "fleiss_kappa_summary.txt"
    with open(out_fleiss, "w", encoding="utf-8") as f:
        f.write(f"Sentence-level rows: {N_sent}\n")
        f.write(f"Sentence-level polarity-only rows (Car!=Neu & NLT!=Neu): {N_sent_pol}\n")
        f.write(f"Fleiss kappa (3-way) [Sentence]: {fleiss_sent_3:.6f}\n")
        f.write(f"Fleiss kappa (polarity-only) [Sentence]: {fleiss_sent_2:.6f}\n\n")

        f.write(f"Utterance-level rows: {N_utt}\n")
        f.write(f"Utterance-level polarity-only rows (Car!=Neu & NLT!=Neu): {N_utt_pol}\n")
        f.write(f"Fleiss kappa (3-way) [Utterance]: {fleiss_utt_3:.6f}\n")
        f.write(f"Fleiss kappa (polarity-only) [Utterance]: {fleiss_utt_2:.6f}\n")

    out_latex = out_dir / "kappa_table_rows.tex"
    with open(out_latex, "w", encoding="utf-8") as f:
        def latex_pair(a, b):
            return f"{PRETTY[a]}--{PRETTY[b]}"

        for level in ["Utterance", "Sentence"]:
            sub = summary_fmt[summary_fmt["Level"] == level].copy()
            sub["pair_order"] = sub["Pair"].map({"SiE--Car": 0, "SiE--NLT": 1, "Car--NLT": 2})
            sub = sub.sort_values("pair_order")

            f.write(f"% {level} rows\n")
            for _, r in sub.iterrows():
                a, b = r["Pair"].split("--")
                f.write(
                    f"& {latex_pair(a, b)} & {int(r['N']):,} & {r['Agr']:.1f} & {r['kappa']:.3f} "
                    f"& {int(r['N_neg0']):,} & {r['Agr_neg0']:.1f} & {r['kappa_neg0']:.3f} \\\n"
                )
            f.write("\n")

        f.write("% Fleiss kappas\n")
        f.write(f"% Fleiss kappa (3-way): Sentence={fleiss_sent_3:.3f}, Utterance={fleiss_utt_3:.3f}\n")
        f.write(
            f"% Fleiss kappa (polarity-only): Sentence={fleiss_sent_2:.3f}, Utterance={fleiss_utt_2:.3f}\n"
        )

    print("DONE.")
    print(f"Outputs saved to: {out_dir}")
    print(f"- {out_summary_csv.name}")
    print(f"- {out_fleiss.name}")
    print(f"- {out_latex.name}")
    print("- confusion_*_raw.csv and confusion_*_row_normalized.csv (for each pair and level)")
    print()
    print("Fleiss (3-way): Sentence =", round(fleiss_sent_3, 3), "Utterance =", round(fleiss_utt_3, 3))
    print("Fleiss (polarity-only): Sentence =", round(fleiss_sent_2, 3), "Utterance =", round(fleiss_utt_2, 3))


if __name__ == "__main__":
    main()
