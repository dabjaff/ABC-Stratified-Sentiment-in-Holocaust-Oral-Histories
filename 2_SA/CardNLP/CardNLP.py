import csv
import logging
import os
import re
from collections import defaultdict

import nltk
import torch
from nltk.tokenize import sent_tokenize
from transformers import pipeline


# Paths
INPUT_FILE = r"C:?????\1_Parsing Corhoh\answers.csv"
OUTPUT_DIRECTORY = r"C:??????\2_SA\CardNLP"

# CSV columns
ID_COL = "GlobalBlockID"
TEXT_COL = "Text"

# Output files
LOG_FILENAME = "CardiffNlp_Sentiment_Analysis.log"
RESULTS_FILENAME = "Card_all_sentiments.txt"

# Model / pipeline
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABEL_MAPPING = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}

_WHITESPACE_RE = re.compile(r"\s+")


def minimal_normalization(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()



def ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")



def setup_logging(log_path: str) -> None:
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )



def load_processed_ids_from_output(output_path: str) -> set[str]:
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



def ensure_results_header(output_path: str, input_file: str) -> None:
    if os.path.exists(output_path):
        return

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("Sentiment Analysis Results\n")
        handle.write("========================\n\n")
        handle.write(f"File: {os.path.basename(input_file)}\n")
        handle.write("=" * 50 + "\n")



def validate_input_file(input_file: str) -> None:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found:\n{input_file}")



def validate_columns(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise ValueError("CSV has no header row / fieldnames.")

    required = {ID_COL, TEXT_COL}
    found = set(fieldnames)
    if not required.issubset(found):
        raise ValueError(
            f"CSV is missing required columns.\n"
            f"Found: {fieldnames}\n"
            f"Required: {sorted(required)}"
        )



def build_sentiment_analyzer():
    device = 0 if torch.cuda.is_available() else -1
    logging.info("Device: %s", "cuda" if device == 0 else "cpu")
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device,
        truncation=True,
        max_length=512,
    )



def analyze_sentences(sentences: list[str], analyzer):
    label_counts = defaultdict(int)
    label_confidences = defaultdict(list)
    sentence_results = []
    sentence_logs = []

    for sent_idx, sentence in enumerate(sentences, start=1):
        normalized_sentence = minimal_normalization(sentence)
        sentiment = analyzer(normalized_sentence)[0]

        raw_label = sentiment.get("label", "")
        label = LABEL_MAPPING.get(raw_label, "Neutral")
        confidence = float(sentiment.get("score", 0.0))

        label_counts[label] += 1
        label_confidences[label].append(confidence)
        sentence_results.append(f"S{sent_idx} ({label}, confidence: {confidence:.2f})")
        sentence_logs.append((sent_idx, label, raw_label, confidence))

    return sentence_results, sentence_logs, label_counts, label_confidences



def aggregate_sentiment(
    label_counts: dict[str, int],
    label_confidences: dict[str, list[float]],
) -> tuple[str, float]:
    total_sentences = sum(label_counts.values())
    if total_sentences == 0:
        return "Neutral", 0.0

    label_scores = {}
    for label, count in label_counts.items():
        percentage = count / total_sentences
        avg_conf = sum(label_confidences[label]) / len(label_confidences[label])
        label_scores[label] = percentage * avg_conf

    aggregated_label = max(label_scores, key=label_scores.get) if label_scores else "Neutral"

    all_confidences = [score for scores in label_confidences.values() for score in scores]
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    return aggregated_label, avg_confidence



def process_csv(input_file: str, output_path: str, analyzer) -> None:
    ensure_results_header(output_path, input_file)
    processed_ids = load_processed_ids_from_output(output_path)
    logging.info("Loaded processed IDs: %d", len(processed_ids))

    with open(input_file, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames)

        with open(output_path, "a", encoding="utf-8") as out:
            for row_idx, row in enumerate(reader, start=1):
                answer_id = (row.get(ID_COL) or "").strip()
                text = (row.get(TEXT_COL) or "").strip()

                if not answer_id:
                    logging.warning("Row %d: Missing %s. Skipping.", row_idx, ID_COL)
                    continue

                if answer_id in processed_ids:
                    continue

                if not text:
                    logging.info('Row %d | Paragraph %s: Empty text. Skipping.', row_idx, answer_id)
                    continue

                sentences = sent_tokenize(text)
                if not sentences:
                    logging.info(
                        'Row %d | Paragraph %s: No sentences after tokenization. Skipping.',
                        row_idx,
                        answer_id,
                    )
                    continue

                sentence_results, sentence_logs, label_counts, label_confidences = analyze_sentences(
                    sentences,
                    analyzer,
                )

                for sent_idx, label, raw_label, confidence in sentence_logs:
                    logging.info(
                        "Paragraph %s | Sentence %d: %s (raw=%s, confidence: %.2f)",
                        answer_id,
                        sent_idx,
                        label,
                        raw_label,
                        confidence,
                    )

                aggregated_label, avg_confidence = aggregate_sentiment(label_counts, label_confidences)

                results_lines = [f'\nParagraph "{answer_id}"']
                results_lines.append(f"Aggregated Sentiment: {aggregated_label}")
                results_lines.append(f"Average Confidence: {avg_confidence:.2f}")
                results_lines.append(f"consists of {len(sentences)} sentences.")
                results_lines.extend(sentence_results)
                results_lines.append(f'END Paragraph "{answer_id}"')

                out.write("\n".join(results_lines) + "\n")
                out.flush()



def main() -> None:
    validate_input_file(INPUT_FILE)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIRECTORY, LOG_FILENAME)
    output_path = os.path.join(OUTPUT_DIRECTORY, RESULTS_FILENAME)

    setup_logging(log_path)
    logging.info("=== CardiffNLP Script start ===")
    logging.info("Input file: %s", INPUT_FILE)
    logging.info("Output directory: %s", OUTPUT_DIRECTORY)

    ensure_nltk_punkt()
    analyzer = build_sentiment_analyzer()

    try:
        process_csv(INPUT_FILE, output_path, analyzer)
        logging.info("Results written to: %s", output_path)
        logging.info("=== CardiffNLP Script completed successfully ===")
    except Exception as exc:
        logging.error("Error processing %s: %s", INPUT_FILE, exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
