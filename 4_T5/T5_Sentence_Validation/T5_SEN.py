import os
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


INPUT_PATH = r"C:\Users\so86qih\Downloads\Third Round\Third Round\3_Parsing the log files\sentence_level_abc_details.csv"
OUTPUT_DIR = r"C:\Users\so86qih\Downloads\Third Round\Third Round\New folder\T5_Sentence_Validation"

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "t5_sentence_group4_heatmap_results.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "t5_group4_heatmap.log")

SAMPLE_PER_GROUP = 1000
MIN_WORDS = 10
MAX_WORDS = 350

MODEL_NAME = "mrm8488/t5-base-finetuned-emotion"
BATCH_SIZE = 8
MAX_INPUT_LEN = 512
MAX_NEW_TOKENS = 10

GROUP_ORDER = ["A_-1", "A_+1", "B", "C"]


def setup_logger() -> logging.Logger:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def label_to_pol(avg_sentiment) -> int | None:
    s = str(avg_sentiment).strip().lower()
    if s.startswith("pos"):
        return 1
    if s.startswith("neg"):
        return -1
    if s.startswith("neu"):
        return 0
    return None


def assign_group(sentence_abc: str, polarity: int | None) -> str | None:
    cat = str(sentence_abc).strip()

    if polarity is None:
        return None
    if cat == "A":
        if polarity == 1:
            return "A_+1"
        if polarity == -1:
            return "A_-1"
        return None
    if cat == "B":
        return "B"
    if cat == "C":
        return "C"
    return None


def run_t5_with_confidence(texts, tokenizer, model, device):
    results = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Analyzing Sentences"):
        batch = [str(t) for t in texts[i : i + BATCH_SIZE]]
        inputs = tokenizer(
            [f"classify emotion: {t}" for t in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            scores = model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
            )

        for j in range(len(batch)):
            label = tokenizer.decode(outputs.sequences[j], skip_special_tokens=True)
            if scores[j].numel() > 0:
                conf = round(torch.exp(scores[j][0]).item(), 4)
            else:
                conf = float("nan")
            results.append({"t5_label": label, "t5_confidence": conf})

    return results


def load_and_sample(logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading input CSV: %s", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    required_cols = {"SentenceText", "Avg_Sentiment", "Sentence_ABC_Category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

    df["SentenceText"] = df["SentenceText"].astype(str)
    df["Avg_Sentiment"] = df["Avg_Sentiment"].astype(str).str.strip()
    df["Sentence_ABC_Category"] = df["Sentence_ABC_Category"].astype(str).str.strip()

    df["word_count"] = df["SentenceText"].apply(lambda x: len(x.split()))
    filtered_df = df[(df["word_count"] >= MIN_WORDS) & (df["word_count"] <= MAX_WORDS)].copy()
    filtered_df["SentencePolarity"] = filtered_df["Avg_Sentiment"].apply(label_to_pol)
    filtered_df["AnalysisGroup"] = filtered_df.apply(
        lambda row: assign_group(row["Sentence_ABC_Category"], row["SentencePolarity"]),
        axis=1,
    )
    filtered_df = filtered_df[filtered_df["AnalysisGroup"].notna()].copy()

    available_counts = filtered_df["AnalysisGroup"].value_counts().to_dict()
    logger.info("Available after filtering: %s", available_counts)

    sample_df = (
        filtered_df.groupby("AnalysisGroup", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), SAMPLE_PER_GROUP), random_state=42))
        .reset_index(drop=True)
    )
    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

    sample_counts = sample_df["AnalysisGroup"].value_counts().to_dict()
    logger.info("Sampled %d sentences across groups: %s", len(sample_df), sample_counts)
    logger.info(
        "T5 will examine %d sentences (max possible = %d).",
        len(sample_df),
        4 * SAMPLE_PER_GROUP,
    )

    missing_groups = [group for group in GROUP_ORDER if group not in sample_counts]
    if missing_groups:
        logger.warning("Some groups are missing (this is OK): %s", missing_groups)

    return sample_df


def build_heatmaps(final_df: pd.DataFrame, logger: logging.Logger) -> None:
    try:
        pivot_dist = pd.crosstab(final_df["AnalysisGroup"], final_df["t5_label"], normalize="index")
        pivot_dist = pivot_dist.reindex([group for group in GROUP_ORDER if group in pivot_dist.index])

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_dist, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Emotion Frequency Heatmap (Distribution %)")
        plt.tight_layout()
        out1 = os.path.join(OUTPUT_DIR, "1_emotion_distribution_group4.png")
        plt.savefig(out1, dpi=200)
        plt.close()

        pivot_conf = final_df.pivot_table(
            index="AnalysisGroup",
            columns="t5_label",
            values="t5_confidence",
            aggfunc="mean",
        )
        pivot_conf = pivot_conf.reindex([group for group in GROUP_ORDER if group in pivot_conf.index])

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_conf, annot=True, cmap="mako", fmt=".2f")
        plt.title("T5 Confidence Heatmap (Average Certainty Score)")
        plt.tight_layout()
        out2 = os.path.join(OUTPUT_DIR, "2_model_confidence_group4.png")
        plt.savefig(out2, dpi=200)
        plt.close()

        logger.info("Saved heatmaps:\n- %s\n- %s", out1, out2)
    except Exception as exc:
        logger.warning("Visualization error: %s", exc)


def main() -> None:
    logger = setup_logger()

    sample_df = load_and_sample(logger)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    logger.info("Starting T5 emotion inference...")
    results_data = run_t5_with_confidence(sample_df["SentenceText"].tolist(), tokenizer, model, device)

    final_df = pd.concat([sample_df.reset_index(drop=True), pd.DataFrame(results_data)], axis=1)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    logger.info("Saved results CSV: %s", OUTPUT_FILE)

    build_heatmaps(final_df, logger)
    logger.info("=== TASK COMPLETE ===")


if __name__ == "__main__":
    main()
