from __future__ import annotations

import argparse
import csv
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_ABBR = ["Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "vs", "etc", "e.g", "i.e"]
_ABBR_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBR) + r")\.", flags=re.IGNORECASE)
_INITIAL_RE = re.compile(r"\b([A-Z])\.", flags=re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+", flags=re.UNICODE)


def _try_nltk_sent_tokenize(text: str) -> Optional[List[str]]:
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore

        sents = sent_tokenize(text)
        return [s.strip() for s in sents if s and s.strip()]
    except Exception:
        return None


def split_sentences(text: str) -> List[str]:
    if text is None:
        return []
    t = str(text).strip()
    if not t:
        return []
    t = re.sub(r"\s+", " ", t)

    nltk_sents = _try_nltk_sent_tokenize(t)
    if nltk_sents:
        return nltk_sents

    t = t.replace("...", "<ELLIPSIS>")
    t = _ABBR_RE.sub(lambda m: m.group(1) + "<DOT>", t)
    t = _INITIAL_RE.sub(lambda m: m.group(1) + "<DOT>", t)

    parts = _SENT_SPLIT_RE.split(t)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p.replace("<DOT>", ".").replace("<ELLIPSIS>", "..."))
    return out


def load_answers_map(path: Path, logger: logging.Logger) -> Dict[str, str]:
    if not path.exists():
        logger.warning(f"answers.csv not found at: {path}. Text columns will be blank.")
        return {}

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("answers.csv appears empty (no header row).")

        cols = set(reader.fieldnames)
        if "GlobalBlockID" not in cols:
            raise ValueError(f"answers.csv must contain GlobalBlockID. Found: {sorted(cols)}")

        text_col = "UTTE_Text" if "UTTE_Text" in cols else ("Text" if "Text" in cols else None)
        if not text_col:
            raise ValueError(f"answers.csv must contain UTTE_Text or Text. Found: {sorted(cols)}")

        out: Dict[str, str] = {}
        dup = 0
        for row in reader:
            pid = (row.get("GlobalBlockID") or "").strip()
            if not pid or pid.lower() == "nan":
                continue
            if pid in out:
                dup += 1
                continue
            out[pid] = (row.get(text_col) or "").strip()
        if dup:
            logger.warning(f"answers.csv has {dup} duplicate GlobalBlockID rows (kept first occurrence).")
        logger.info(f"Loaded answers.csv text map: {len(out)} utterances.")
        return out


PARA_RE = re.compile(r'^Paragraph\s+"(?P<pid>[^"]+)"\s*$')
AGG_RE = re.compile(r'^Aggregated\s+Sentiment:\s*(?P<label>\w+)\s*$')
CONF_RE = re.compile(r'^Average\s+Confidence:\s*(?P<conf>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$')
COUNT_RE = re.compile(r'^consists\s+of\s+(?P<count>\d+)\s+sentences\.\s*$')
SENT_RE = re.compile(
    r'^S(?P<sidx>\d+)\s*\(\s*(?P<label>Negative|Neutral|Positive)\s*,\s*confidence:\s*(?P<conf>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)\s*$',
    re.IGNORECASE,
)


def norm_label(label: str) -> str:
    l = (label or "").strip().lower()
    if l.startswith("neg"):
        return "Negative"
    if l.startswith("neu"):
        return "Neutral"
    if l.startswith("pos"):
        return "Positive"
    return (label or "").strip().title() if label else ""


def safe_float(x: Optional[str]) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class ParagraphBlock:
    pid: str
    agg_label: str
    avg_conf: float
    sent_count: int
    sentences: List[Tuple[int, str, float]]


def parse_sentiment_file(path: Path, logger: logging.Logger) -> List[ParagraphBlock]:
    blocks: List[ParagraphBlock] = []

    cur_pid: Optional[str] = None
    cur_agg: Optional[str] = None
    cur_avg_conf: float = float("nan")
    cur_count: Optional[int] = None
    cur_sents: List[Tuple[int, str, float]] = []

    def reset() -> None:
        nonlocal cur_pid, cur_agg, cur_avg_conf, cur_count, cur_sents
        cur_pid, cur_agg, cur_avg_conf, cur_count, cur_sents = None, None, float("nan"), None, []

    def emit_if_ready(reason: str) -> None:
        nonlocal cur_pid, cur_agg, cur_avg_conf, cur_count, cur_sents
        if cur_pid is None:
            return
        if cur_count is None:
            logger.warning(
                f"{path.name}: Paragraph {cur_pid} missing sentence-count line before emit ({reason}). Skipping block."
            )
            reset()
            return

        if len(cur_sents) != cur_count:
            logger.warning(
                f"{path.name}: Paragraph {cur_pid} sentence lines count mismatch: expected {cur_count}, found {len(cur_sents)}."
            )

        blocks.append(
            ParagraphBlock(
                pid=cur_pid,
                agg_label=norm_label(cur_agg or ""),
                avg_conf=cur_avg_conf,
                sent_count=cur_count,
                sentences=cur_sents.copy(),
            )
        )
        reset()

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")

            m = PARA_RE.match(line)
            if m:
                if cur_pid is not None:
                    emit_if_ready(f"new paragraph at line {line_no}")
                cur_pid = m.group("pid")
                continue

            if cur_pid is None:
                continue

            m = AGG_RE.match(line)
            if m:
                cur_agg = m.group("label")
                continue

            m = CONF_RE.match(line)
            if m:
                cur_avg_conf = safe_float(m.group("conf"))
                continue

            m = COUNT_RE.match(line)
            if m:
                cur_count = int(m.group("count"))
                continue

            m = SENT_RE.match(line)
            if m:
                sidx = int(m.group("sidx"))
                lab = norm_label(m.group("label"))
                conf = safe_float(m.group("conf"))
                cur_sents.append((sidx, lab, conf))
                continue

        if cur_pid is not None:
            emit_if_ready("end of file")

    return blocks


def count_labels(labels: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {"Negative": 0, "Neutral": 0, "Positive": 0}
    for l in labels:
        l2 = norm_label(l)
        if l2 in out:
            out[l2] += 1
    return out


def fmt_counts(d: Dict[str, int]) -> str:
    parts = []
    if d.get("Negative", 0):
        parts.append(f"Neg {d['Negative']}")
    if d.get("Positive", 0):
        parts.append(f"Pos {d['Positive']}")
    if d.get("Neutral", 0):
        parts.append(f"Neu {d['Neutral']}")
    return ", ".join(parts) if parts else ""


def vote_sentence(
    sbert: Tuple[str, float],
    card: Tuple[str, float],
    ntown: Tuple[str, float],
) -> str:
    lab_s, conf_s = sbert
    lab_c, conf_c = card
    lab_n, conf_n = ntown

    labs = [norm_label(lab_s), norm_label(lab_c), norm_label(lab_n)]
    counts = count_labels(labs)

    for lab in ("Negative", "Neutral", "Positive"):
        if counts.get(lab, 0) >= 2:
            return lab

    confs = [
        (conf_s if not math.isnan(conf_s) else float("-inf"), labs[0]),
        (conf_c if not math.isnan(conf_c) else float("-inf"), labs[1]),
        (conf_n if not math.isnan(conf_n) else float("-inf"), labs[2]),
    ]
    confs.sort(key=lambda x: x[0], reverse=True)
    top_conf, top_lab = confs[0]

    if top_conf == float("-inf"):
        return labs[0]
    if len(confs) >= 2 and confs[1][0] == top_conf:
        return labs[0]
    return top_lab


def inter_model_agreement(lab_s: str, lab_c: str, lab_n: str) -> str:
    a = norm_label(lab_s)
    b = norm_label(lab_c)
    c = norm_label(lab_n)

    if a == b == c:
        return "3/3"
    if a == b or a == c or b == c:
        return "2/3"
    return "1/3"


def sentence_distribution(block: ParagraphBlock) -> Dict[str, int]:
    labels = [lab for _, lab, _ in block.sentences]
    return count_labels(labels)


def build_avg_sentence_counts(
    sbert: ParagraphBlock,
    card: ParagraphBlock,
    ntown: ParagraphBlock,
    logger: logging.Logger,
) -> Dict[str, int]:
    n = min(sbert.sent_count, card.sent_count, ntown.sent_count)
    sb_map = {i: (lab, conf) for i, lab, conf in sbert.sentences}
    cd_map = {i: (lab, conf) for i, lab, conf in card.sentences}
    nt_map = {i: (lab, conf) for i, lab, conf in ntown.sentences}

    out = {"Negative": 0, "Neutral": 0, "Positive": 0}
    for i in range(1, n + 1):
        if i not in sb_map or i not in cd_map or i not in nt_map:
            logger.warning(
                f"Missing sentence {i} in paragraph {sbert.pid}: "
                f"SBert={i in sb_map}, Card={i in cd_map}, NLPTown={i in nt_map}. "
                "Skipping vote for this sentence."
            )
            continue
        voted = vote_sentence(sb_map[i], cd_map[i], nt_map[i])
        out[voted] += 1
    return out


ABC_SET = {"A", "B", "C"}


def abc_category(lab_s: str, lab_c: str, lab_n: str) -> Optional[str]:
    a = norm_label(lab_s)
    b = norm_label(lab_c)
    c = norm_label(lab_n)
    labels = [a, b, c]
    uniq = set(labels)

    if len(uniq) == 3:
        return "C"
    if a == b == c and a in {"Positive", "Negative"}:
        return "A"
    if b == "Neutral" or c == "Neutral":
        return "B"

    pos = labels.count("Positive")
    neg = labels.count("Negative")
    if pos >= 2 or neg >= 2:
        return "B"

    return None


def utterance_polarity_num(lab_s: str, lab_c: str, lab_n: str) -> str:
    labs = [norm_label(lab_s), norm_label(lab_c), norm_label(lab_n)]
    if labs.count("Positive") >= 2:
        return "1"
    if labs.count("Negative") >= 2:
        return "-1"
    if labs.count("Neutral") >= 2:
        return "0"

    sb = labs[0]
    if sb == "Positive":
        return "1"
    if sb == "Negative":
        return "-1"
    if sb == "Neutral":
        return "0"
    return "0"


def setup_logger(outdir: Path) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "merge_paragraph_labels.log"

    logger = logging.getLogger("merge_paragraph_labels")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    default_sbert = project_dir / "2_SA" / "SBert" / "S_all_sentiments.txt"
    default_card = project_dir / "2_SA" / "CardNLP" / "Card_all_sentiments.txt"
    default_ntown = project_dir / "2_SA" / "NLPT" / "NTown_all_sentiments.txt"
    default_answers = project_dir / "1_Parsing Corhoh" / "answers.csv"
    default_outdir = script_dir

    parser = argparse.ArgumentParser(description="Merge paragraph sentiment files and write the merged exports.")
    parser.add_argument("--sbert", type=str, default=str(default_sbert), help="Path to the SBert TXT file")
    parser.add_argument("--card", type=str, default=str(default_card), help="Path to the Card TXT file")
    parser.add_argument("--ntown", type=str, default=str(default_ntown), help="Path to the NLPTown TXT file")
    parser.add_argument("--answers", type=str, default=str(default_answers), help="Path to answers.csv")
    parser.add_argument("--outdir", type=str, default=str(default_outdir), help="Output folder")
    parser.add_argument("--strict", action="store_true", help="Stop on first structural ID mismatch")
    args = parser.parse_args()

    SBERT_TXT = Path(args.sbert)
    CARD_TXT = Path(args.card)
    NTOWN_TXT = Path(args.ntown)
    ANSWERS_CSV = Path(args.answers)
    OUTDIR = Path(args.outdir)

    logger = setup_logger(OUTDIR)
    logger.info("Starting merge")
    logger.info(f"SBert: {SBERT_TXT}")
    logger.info(f"Card:  {CARD_TXT}")
    logger.info(f"NLPT:  {NTOWN_TXT}")
    logger.info(f"Answers: {ANSWERS_CSV}")
    logger.info(f"Out:   {OUTDIR}")
    logger.info(f"Strict alignment: {bool(args.strict)}")

    answers_map = load_answers_map(ANSWERS_CSV, logger)
    split_cache: Dict[str, List[str]] = {}

    s_blocks = parse_sentiment_file(SBERT_TXT, logger)
    c_blocks = parse_sentiment_file(CARD_TXT, logger)
    n_blocks = parse_sentiment_file(NTOWN_TXT, logger)

    logger.info(f"Parsed blocks: SBert={len(s_blocks)}, Card={len(c_blocks)}, NLPTown={len(n_blocks)}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pos_path = OUTDIR / "all_three_positive.csv"
    neg_path = OUTDIR / "all_three_negative.csv"
    dis_path = OUTDIR / "disagreed.csv"
    sum_path = OUTDIR / "sentence_level_summary.csv"
    det_path = OUTDIR / "sentence_level_details.csv"

    utt_abc_path = OUTDIR / "utterance_level_abc.csv"
    utt_abc_counts_path = OUTDIR / "utterance_level_abc_counts.csv"
    sent_abc_path = OUTDIR / "sentence_level_abc_details.csv"
    sent_abc_counts_path = OUTDIR / "sentence_level_abc_counts.csv"
    utt_from_sent_path = OUTDIR / "utterance_summary_from_sentence_abc.csv"
    master_sheet_path = OUTDIR / "master_sheet_sentence_utterance.csv"

    base_fields = [
        "ParagraphID",
        "SentenceCount",
        "SBert_AggregatedSentiment",
        "SBert_AverageConfidence",
        "Card_AggregatedSentiment",
        "Card_AverageConfidence",
        "NLPTown_AggregatedSentiment",
        "NLPTown_AverageConfidence",
    ]
    summary_fields = base_fields + ["SBert_S_S", "Card_S_S", "NLPTown_S_S", "Avg_S"]

    detail_fields = [
        "ParagraphID",
        "SentenceIndex",
        "SBert_Sentiment",
        "SBert_Confidence",
        "Card_Sentiment",
        "Card_Confidence",
        "NLPTown_Sentiment",
        "NLPTown_Confidence",
        "InterModelAgreement",
        "Avg_Sentiment",
        "SentenceText",
    ]

    utt_abc_fields = base_fields + ["Utterance_ABC_Category", "UtterancePolarity", "UTTE_Text"]
    sent_abc_fields = detail_fields + ["Sentence_ABC_Category"]
    master_sheet_fields = [
        "ParagraphID",
        "UtterancePolarity",
        "Utterance_ABC_Category",
        "UTTE_Text",
        "SentenceIndex",
        "SentenceText",
        "Avg_Sentiment",
        "Sentence_ABC_Category",
    ]

    utt_abc_counts: Dict[str, int] = {k: 0 for k in ABC_SET}
    sent_abc_counts: Dict[str, int] = {k: 0 for k in ABC_SET}
    per_paragraph_sentcat: Dict[str, Dict[str, int]] = {}

    with (
        pos_path.open("w", newline="", encoding="utf-8") as pos_f,
        neg_path.open("w", newline="", encoding="utf-8") as neg_f,
        dis_path.open("w", newline="", encoding="utf-8") as dis_f,
        sum_path.open("w", newline="", encoding="utf-8") as sum_f,
        det_path.open("w", newline="", encoding="utf-8") as det_f,
        utt_abc_path.open("w", newline="", encoding="utf-8") as utt_f,
        sent_abc_path.open("w", newline="", encoding="utf-8") as sent_f,
        master_sheet_path.open("w", newline="", encoding="utf-8") as master_f,
    ):
        pos_w = csv.DictWriter(pos_f, fieldnames=summary_fields)
        neg_w = csv.DictWriter(neg_f, fieldnames=summary_fields)
        dis_w = csv.DictWriter(dis_f, fieldnames=summary_fields)
        sum_w = csv.DictWriter(sum_f, fieldnames=summary_fields)
        det_w = csv.DictWriter(det_f, fieldnames=detail_fields)

        utt_w = csv.DictWriter(utt_f, fieldnames=utt_abc_fields)
        sent_w = csv.DictWriter(sent_f, fieldnames=sent_abc_fields)
        master_w = csv.DictWriter(master_f, fieldnames=master_sheet_fields)

        for w in (pos_w, neg_w, dis_w, sum_w):
            w.writeheader()
        det_w.writeheader()
        utt_w.writeheader()
        sent_w.writeheader()
        master_w.writeheader()

        total_rows = min(len(s_blocks), len(c_blocks), len(n_blocks))
        id_mismatch = 0
        count_mismatch = 0
        skipped_utt_none = 0
        skipped_sent_none = 0

        for idx in range(total_rows):
            sb = s_blocks[idx]
            cd = c_blocks[idx]
            nt = n_blocks[idx]

            if not (sb.pid == nt.pid == cd.pid):
                id_mismatch += 1
                logger.error(f"ID mismatch at row {idx + 1}: SBert={sb.pid}, Card={cd.pid}, NLPTown={nt.pid}")
                if args.strict:
                    raise RuntimeError("Structural mismatch: paragraph IDs diverged. Aborting (strict mode).")

            if not (sb.sent_count == nt.sent_count == cd.sent_count):
                count_mismatch += 1
                logger.warning(
                    f"Sentence-count mismatch for {sb.pid}/{nt.pid}/{cd.pid} at row {idx + 1}: "
                    f"SBert={sb.sent_count}, NTown={nt.sent_count}, Card={cd.sent_count}"
                )

            sb_dist = sentence_distribution(sb)
            cd_dist = sentence_distribution(cd)
            nt_dist = sentence_distribution(nt)
            avg_dist = build_avg_sentence_counts(sb, cd, nt, logger)

            row = {
                "ParagraphID": sb.pid,
                "SentenceCount": sb.sent_count,
                "SBert_AggregatedSentiment": sb.agg_label,
                "SBert_AverageConfidence": f"{sb.avg_conf:.6f}" if not math.isnan(sb.avg_conf) else "nan",
                "Card_AggregatedSentiment": cd.agg_label,
                "Card_AverageConfidence": f"{cd.avg_conf:.6f}" if not math.isnan(cd.avg_conf) else "nan",
                "NLPTown_AggregatedSentiment": nt.agg_label,
                "NLPTown_AverageConfidence": f"{nt.avg_conf:.6f}" if not math.isnan(nt.avg_conf) else "nan",
                "SBert_S_S": fmt_counts(sb_dist),
                "Card_S_S": fmt_counts(cd_dist),
                "NLPTown_S_S": fmt_counts(nt_dist),
                "Avg_S": fmt_counts(avg_dist),
            }

            sum_w.writerow(row)

            if sb.agg_label == "Positive" and cd.agg_label == "Positive" and nt.agg_label == "Positive":
                pos_w.writerow(row)
            elif sb.agg_label == "Negative" and cd.agg_label == "Negative" and nt.agg_label == "Negative":
                neg_w.writerow(row)
            else:
                dis_w.writerow(row)

            utt_cat = abc_category(sb.agg_label, cd.agg_label, nt.agg_label)
            utt_pol = ""
            if utt_cat is None:
                skipped_utt_none += 1
            else:
                utt_abc_counts[utt_cat] += 1
                utt_pol = utterance_polarity_num(sb.agg_label, cd.agg_label, nt.agg_label)
                utt_w.writerow(
                    {
                        **{k: row[k] for k in base_fields},
                        "Utterance_ABC_Category": utt_cat,
                        "UtterancePolarity": utt_pol,
                        "UTTE_Text": answers_map.get(sb.pid, ""),
                    }
                )

            sb_map = {i: (lab, conf) for i, lab, conf in sb.sentences}
            cd_map = {i: (lab, conf) for i, lab, conf in cd.sentences}
            nt_map = {i: (lab, conf) for i, lab, conf in nt.sentences}
            n = min(sb.sent_count, cd.sent_count, nt.sent_count)

            if sb.pid not in per_paragraph_sentcat:
                per_paragraph_sentcat[sb.pid] = {k: 0 for k in ABC_SET}

            for sidx in range(1, n + 1):
                if sidx not in sb_map or sidx not in cd_map or sidx not in nt_map:
                    continue
                sb_lab, sb_conf = sb_map[sidx]
                cd_lab, cd_conf = cd_map[sidx]
                nt_lab, nt_conf = nt_map[sidx]
                avg_lab = vote_sentence((sb_lab, sb_conf), (cd_lab, cd_conf), (nt_lab, nt_conf))

                sent_text = ""
                if answers_map:
                    if sb.pid not in split_cache:
                        split_cache[sb.pid] = split_sentences(answers_map.get(sb.pid, ""))
                    sents_txt = split_cache.get(sb.pid, [])
                    if 1 <= sidx <= len(sents_txt):
                        sent_text = sents_txt[sidx - 1]
                    else:
                        logger.warning(f"SentenceIndex out of range for {sb.pid}: {sidx} > {len(sents_txt)}")

                det_row = {
                    "ParagraphID": sb.pid,
                    "SentenceIndex": sidx,
                    "SBert_Sentiment": norm_label(sb_lab),
                    "SBert_Confidence": f"{sb_conf:.6f}" if not math.isnan(sb_conf) else "nan",
                    "Card_Sentiment": norm_label(cd_lab),
                    "Card_Confidence": f"{cd_conf:.6f}" if not math.isnan(cd_conf) else "nan",
                    "NLPTown_Sentiment": norm_label(nt_lab),
                    "NLPTown_Confidence": f"{nt_conf:.6f}" if not math.isnan(nt_conf) else "nan",
                    "InterModelAgreement": inter_model_agreement(sb_lab, cd_lab, nt_lab),
                    "Avg_Sentiment": avg_lab,
                    "SentenceText": sent_text,
                }
                det_w.writerow(det_row)

                sent_cat = abc_category(sb_lab, cd_lab, nt_lab)

                master_w.writerow(
                    {
                        "ParagraphID": sb.pid,
                        "UtterancePolarity": utt_pol if sidx == 1 else "",
                        "Utterance_ABC_Category": (utt_cat or "") if sidx == 1 else "",
                        "UTTE_Text": answers_map.get(sb.pid, "") if sidx == 1 else "",
                        "SentenceIndex": sidx,
                        "SentenceText": sent_text,
                        "Avg_Sentiment": avg_lab,
                        "Sentence_ABC_Category": sent_cat or "",
                    }
                )

                if sent_cat is None:
                    skipped_sent_none += 1
                    continue

                sent_abc_counts[sent_cat] += 1
                per_paragraph_sentcat[sb.pid][sent_cat] += 1
                sent_w.writerow({**det_row, "Sentence_ABC_Category": sent_cat})

        logger.info(f"Finished. Rows processed: {total_rows}")
        logger.info(f"ID mismatches: {id_mismatch}; sentence-count mismatches: {count_mismatch}")
        logger.info(f"Skipped utterances (unclassifiable): {skipped_utt_none}")
        logger.info(f"Skipped sentences (unclassifiable): {skipped_sent_none}")

    def write_counts_table(outpath: Path, counts: Dict[str, int]) -> None:
        total = sum(counts.values())
        rows = []
        for k in ["A", "B", "C"]:
            v = counts.get(k, 0)
            pct = (v / total * 100) if total else 0.0
            rows.append({"Category": k, "N": v, "Percent": round(pct, 2)})

        with outpath.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Category", "N", "Percent"])
            w.writeheader()
            w.writerows(rows)

    write_counts_table(utt_abc_counts_path, utt_abc_counts)
    write_counts_table(sent_abc_counts_path, sent_abc_counts)

    categories = ["A", "B", "C"]
    with utt_from_sent_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["ParagraphID", "N_sentences_total"] + [f"N_{c}" for c in categories] + [
            f"Pct_{c}" for c in categories
        ] + ["DominantSentenceCategory"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for pid, d in per_paragraph_sentcat.items():
            total = sum(d.get(c, 0) for c in categories)
            row = {"ParagraphID": pid, "N_sentences_total": total}
            for c in categories:
                row[f"N_{c}"] = int(d.get(c, 0))
            for c in categories:
                row[f"Pct_{c}"] = round((row[f"N_{c}"] / total * 100) if total else 0.0, 2)

            dom = max(categories, key=lambda c: (row[f"N_{c}"], -categories.index(c)))
            row["DominantSentenceCategory"] = dom
            w.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
