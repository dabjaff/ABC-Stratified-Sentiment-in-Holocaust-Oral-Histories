import csv
import logging
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import nltk
import torch
from nltk.tokenize import sent_tokenize
from transformers import pipeline


INPUT_FILE = r"C:?????\1_Parsing Corhoh\answers.csv"
OUTPUT_DIRECTORY = r"C:???\2_SA\NLPT"

LOG_FILE = os.path.join(OUTPUT_DIRECTORY, "NlpTown_Sentiment_Analysis.log")
ALL_RESULTS_FILE = os.path.join(OUTPUT_DIRECTORY, "NTown_all_sentiments.txt")
REQUIRED_COLUMNS = {"GlobalBlockID", "Text"}
WHITESPACE_RE = re.compile(r"\s+")


def ensure_nltk_tokenizer() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def map_stars_to_sentiment(label: str) -> str:
    label = label.strip().lower()
    if label.startswith(("1", "2")):
        return "Negative"
    if label.startswith("3"):
        return "Neutral"
    if label.startswith(("4", "5")):
        return "Positive"
    return "Neutral"


def setup_logging() -> None:
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )


def build_sentiment_analyzer():
    device = 0 if torch.cuda.is_available() else -1
    logging.info("Device: %s", "cuda" if device == 0 else "cpu")
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
        device=device,
        truncation=True,
        max_length=512,
    )


def validate_input_file() -> None:
    if not os.path.exists(INPUT_FILE):
        logging.error("Error: Input file '%s' not found.", INPUT_FILE)
        raise FileNotFoundError(
            f"Input file '{INPUT_FILE}' not found. Please check the path."
        )


def load_processed_ids(results_path: str) -> Tuple[Set[str], str]:
    processed_ids: Set[str] = set()

    if not os.path.exists(results_path):
        return processed_ids, "w"

    with open(results_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith('Paragraph "'):
                continue
            parts = line.split('"')
            if len(parts) >= 2:
                processed_ids.add(parts[1])

    return processed_ids, "a"


def initialize_results_file(results_path: str) -> None:
    with open(results_path, "w", encoding="utf-8") as handle:
        handle.write("Sentiment Analysis Results\n========================\n\n")
        handle.write(f"File: {os.path.basename(INPUT_FILE)}\n" + "=" * 50 + "\n")


def validate_columns(fieldnames: Iterable[str] | None) -> None:
    fields = set(fieldnames or [])
    if not REQUIRED_COLUMNS.issubset(fields):
        raise ValueError(
            f"CSV is missing required columns. Found: {list(fieldnames or [])}. "
            f"Required: {sorted(REQUIRED_COLUMNS)}"
        )


def analyze_paragraph(
    answer_id: str,
    text: str,
    sentiment_analyzer,
) -> List[str]:
    sentences = sent_tokenize(text)
    results = [f'\nParagraph "{answer_id}"']

    label_counts: Dict[str, int] = defaultdict(int)
    label_confidences: Dict[str, List[float]] = defaultdict(list)
    sentence_results: List[str] = []

    for sent_idx, sentence in enumerate(sentences, start=1):
        normalized_sentence = normalize_text(sentence)
        sentiment = sentiment_analyzer(normalized_sentence)

        raw_label = sentiment[0]["label"]
        mapped_label = map_stars_to_sentiment(raw_label)
        confidence = sentiment[0]["score"]

        label_counts[mapped_label] += 1
        label_confidences[mapped_label].append(confidence)
        sentence_results.append(
            f"S{sent_idx} ({mapped_label}, confidence: {confidence:.2f})"
        )

        logging.info(
            "File: %s | Paragraph %s | Sentence %s: %s (raw=%s, confidence: %.2f)",
            INPUT_FILE,
            answer_id,
            sent_idx,
            mapped_label,
            raw_label,
            confidence,
        )

    total_sentences = sum(label_counts.values())
    label_scores: Dict[str, float] = {}
    for label, count in label_counts.items():
        percentage = count / total_sentences
        avg_conf = sum(label_confidences[label]) / len(label_confidences[label])
        label_scores[label] = percentage * avg_conf

    aggregated_label = max(label_scores, key=label_scores.get) if label_scores else "Neutral"
    avg_confidence = (
        sum(conf for conf_list in label_confidences.values() for conf in conf_list)
        / total_sentences
        if total_sentences
        else 0
    )

    results.append(f"Aggregated Sentiment: {aggregated_label}")
    results.append(f"Average Confidence: {avg_confidence:.2f}")
    results.append(f"consists of {len(sentences)} sentences.")
    results.extend(sentence_results)
    return results


def process_file() -> None:
    processed_ids, results_mode = load_processed_ids(ALL_RESULTS_FILE)
    if results_mode == "w":
        initialize_results_file(ALL_RESULTS_FILE)

    sentiment_analyzer = build_sentiment_analyzer()

    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames)

        for row_idx, row in enumerate(reader, start=1):
            answer_id = (row.get("GlobalBlockID") or "").strip()
            text = (row.get("Text") or "").strip()

            if not answer_id:
                logging.warning("Row %s: Missing GlobalBlockID. Skipping.", row_idx)
                continue

            if answer_id in processed_ids:
                continue

            if not text:
                logging.info(
                    'Row %s | Paragraph %s: Empty text. Skipping.',
                    row_idx,
                    answer_id,
                )
                continue

            results = analyze_paragraph(answer_id, text, sentiment_analyzer)
            with open(ALL_RESULTS_FILE, "a", encoding="utf-8") as out_handle:
                out_handle.write("\n".join(results) + "\n")


def main() -> None:
    ensure_nltk_tokenizer()
    setup_logging()

    logging.info("NlpTown sentiment analysis started.")
    logging.info("Input file: %s", INPUT_FILE)
    logging.info("Output directory: %s", OUTPUT_DIRECTORY)

    validate_input_file()
    process_file()
    logging.info("NlpTown sentiment analysis completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("Error processing %s: %s", INPUT_FILE, exc)
        raise
