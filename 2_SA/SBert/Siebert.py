import csv
import logging
import os
import re
from collections import defaultdict

import nltk
import torch
from nltk.tokenize import sent_tokenize
from transformers import pipeline


INPUT_FILE = r"C:??????\1_Parsing Corhoh\answers.csv"
OUTPUT_DIRECTORY = r"C:??????\SBert"
MODEL_NAME = "siebert/sentiment-roberta-large-english"

LOG_FILENAME = "Siebert_Sentiment_Analysis.log"
RESULTS_FILENAME = "S_all_sentiments.txt"
REQUIRED_COLUMNS = {"GlobalBlockID", "Text"}
LABEL_MAPPING = {
    "POSITIVE": "Positive",
    "NEGATIVE": "Negative",
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_sentences(sentences):
    normalized = []
    for sentence in sentences:
        sentence = sentence.replace("\n", " ").replace("\t", " ")
        sentence = _WHITESPACE_RE.sub(" ", sentence).strip()
        normalized.append(sentence)
    return normalized


def load_processed_ids(output_path: str) -> set:
    processed = set()
    if not os.path.exists(output_path):
        return processed

    with open(output_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('END Paragraph "'):
                parts = line.split('"')
                if len(parts) >= 2:
                    processed.add(parts[1])
    return processed


def ensure_output_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def configure_logging(log_path: str, input_file: str, output_directory: str) -> None:
    open(log_path, "a", encoding="utf-8").close()
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
        force=True,
    )
    logging.info("=== Script start ===")
    logging.info(f"Input file: {input_file}")
    logging.info(f"Output directory: {output_directory}")



def ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")



def build_sentiment_analyzer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device,
        truncation=True,
        max_length=512,
    )



def write_header_if_needed(output_path: str, input_file: str) -> None:
    open(output_path, "a", encoding="utf-8").close()
    if os.path.getsize(output_path) != 0:
        return

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("Sentiment Analysis Results\n")
        out.write("========================\n\n")
        out.write(f"File: {os.path.basename(input_file)}\n")
        out.write("=" * 50 + "\n")



def analyze_block(block_id: str, text: str, sentiment_analyzer):
    sentences = sent_tokenize(text)
    if not sentences:
        return None, 0, 0.0

    normalized_sentences = normalize_sentences(sentences)
    sentiments = sentiment_analyzer(normalized_sentences)

    label_counts = defaultdict(int)
    label_confidences = defaultdict(list)
    sentence_results = []

    for sent_idx, sentiment in enumerate(sentiments, start=1):
        raw_label = sentiment.get("label", "")
        label = LABEL_MAPPING.get(raw_label, raw_label)
        confidence = float(sentiment.get("score", 0.0))

        label_counts[label] += 1
        label_confidences[label].append(confidence)
        sentence_results.append(f"S{sent_idx} ({label}, confidence: {confidence:.2f})")

    total_sentences = sum(label_counts.values())

    label_scores = {}
    for label in label_counts:
        percentage = label_counts[label] / total_sentences
        avg_conf = sum(label_confidences[label]) / len(label_confidences[label])
        label_scores[label] = percentage * avg_conf

    aggregated_label = max(label_scores, key=label_scores.get) if label_scores else "None"

    all_confidences = [score for scores in label_confidences.values() for score in scores]
    avg_confidence = (
        sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    )

    results = [f'\nParagraph "{block_id}"']
    results.append(f"Aggregated Sentiment: {aggregated_label}")
    results.append(f"Average Confidence: {avg_confidence:.2f}")
    results.append(f"consists of {len(sentences)} sentences.")
    results.extend(sentence_results)
    results.append(f'END Paragraph "{block_id}"')

    return results, len(sentences), avg_confidence



def validate_columns(fieldnames):
    if not fieldnames:
        raise ValueError("CSV has no header row / fieldnames.")
    if not REQUIRED_COLUMNS.issubset(set(fieldnames)):
        raise ValueError(
            f"CSV is missing required columns.\n"
            f"Found: {fieldnames}\n"
            f"Required: {sorted(REQUIRED_COLUMNS)}"
        )



def process_file(input_file: str, output_path: str, sentiment_analyzer) -> None:
    processed_ids = load_processed_ids(output_path)
    logging.info(f"Loaded processed IDs: {len(processed_ids)}")

    with open(input_file, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        validate_columns(reader.fieldnames)

        with open(output_path, "a", encoding="utf-8") as out:
            for row_idx, row in enumerate(reader, start=1):
                block_id = (row.get("GlobalBlockID") or "").strip()
                text = (row.get("Text") or "").strip()

                if not block_id:
                    logging.warning(
                        f"Row {row_idx}: Missing GlobalBlockID. Skipping."
                    )
                    continue

                if block_id in processed_ids:
                    continue

                if not text:
                    logging.info(
                        f'Row {row_idx} | Paragraph {block_id}: Empty text. Skipping.'
                    )
                    continue

                results, sentence_count, avg_confidence = analyze_block(
                    block_id, text, sentiment_analyzer
                )
                if not results:
                    logging.info(
                        f'Row {row_idx} | Paragraph {block_id}: No sentences after tokenization. Skipping.'
                    )
                    continue

                out.write("\n".join(results) + "\n")
                out.flush()

                aggregated_label = results[1].split(": ", 1)[1]
                logging.info(
                    f"Paragraph {block_id} -> Aggregated: {aggregated_label}, "
                    f"Avg confidence: {avg_confidence:.2f}, Sentences: {sentence_count}"
                )

    logging.info(f"Results written to: {output_path}")
    logging.info("=== Script completed successfully ===")



def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    ensure_output_directory(OUTPUT_DIRECTORY)

    log_file = os.path.join(OUTPUT_DIRECTORY, LOG_FILENAME)
    results_file = os.path.join(OUTPUT_DIRECTORY, RESULTS_FILENAME)

    configure_logging(log_file, INPUT_FILE, OUTPUT_DIRECTORY)
    ensure_nltk_punkt()
    write_header_if_needed(results_file, INPUT_FILE)

    sentiment_analyzer = build_sentiment_analyzer()

    try:
        process_file(INPUT_FILE, results_file, sentiment_analyzer)
    except Exception as exc:
        logging.error(f"Error processing {INPUT_FILE}: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
