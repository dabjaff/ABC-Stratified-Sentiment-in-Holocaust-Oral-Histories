import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

INPUT_PATH = r"C:\Users\so86qih\Downloads\Third Round\Third Round\3_Parsing the log files\utterance_level_abc.csv"
OUTPUT_DIR = r"C:\Users\so86qih\Downloads\Third Round\Third Round\New folder\T5_Utterance_Validation"

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "t5_utterance_group4_results.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "t5_utterance_group4.log")

SAMPLE_PER_GROUP = 500
MIN_WORDS = 10
MAX_WORDS = 350

MODEL_NAME = "mrm8488/t5-base-finetuned-emotion"
VALID_LABELS = {"sadness", "joy", "love", "anger", "fear", "surprise"}
GROUP_ORDER = ["A_-1", "A_+1", "B", "C"]


def setup_logger() -> logging.Logger:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def group_logic(row):
    abc = str(row["Utterance_ABC_Category"]).strip()
    pol = row["UtterancePolarity"]

    if abc == "A":
        if pol == 1:
            return "A_+1"
        if pol == -1:
            return "A_-1"
        return None
    if abc == "B":
        return "B"
    if abc == "C":
        return "C"
    return None


def load_and_sample(logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading utterances and building 4 groups: A_+1, A_-1, B, C ...")
    df = pd.read_csv(INPUT_PATH)

    required_cols = ["UTTE_Text", "Utterance_ABC_Category", "UtterancePolarity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in input file: {missing}\nFound columns: {list(df.columns)}"
        )

    df["Utterance_ABC_Category"] = df["Utterance_ABC_Category"].astype(str).str.strip()
    df["UtterancePolarity"] = pd.to_numeric(df["UtterancePolarity"], errors="coerce")
    df["UTTE_Text"] = df["UTTE_Text"].astype(str)

    df["word_count"] = df["UTTE_Text"].apply(lambda x: len(str(x).split()))
    filtered_df = df[(df["word_count"] >= MIN_WORDS) & (df["word_count"] <= MAX_WORDS)].copy()

    filtered_df["AnalysisGroup"] = filtered_df.apply(group_logic, axis=1)
    filtered_df = filtered_df[filtered_df["AnalysisGroup"].notna()].copy()

    sample_df = (
        filtered_df.groupby("AnalysisGroup", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), SAMPLE_PER_GROUP), random_state=42))
        .reset_index(drop=True)
    )
    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

    group_counts = sample_df["AnalysisGroup"].value_counts().to_dict()
    logger.info(f"Sampled {len(sample_df)} UTTERANCES total. Group sizes: {group_counts}")
    logger.info(f"Will examine exactly {len(sample_df)} utterances with T5 (batching below).")

    return sample_df


def load_model(logger: logging.Logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device.upper()}")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    return tokenizer, model, device


def run_t5_with_confidence(texts, tokenizer, model, device):
    results = []
    batch_size = 8

    for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing Utterances"):
        batch = [str(t) for t in texts[i:i + batch_size]]
        inputs = tokenizer(
            [f"classify emotion: {text}" for text in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=10,
            )
            scores = model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
            )

        for j in range(len(batch)):
            label = tokenizer.decode(outputs.sequences[j], skip_special_tokens=True)
            confidence = round(torch.exp(scores[j][0]).item(), 4)
            results.append({"t5_label": label, "t5_confidence": confidence})

    return results


def save_heatmaps(final_df: pd.DataFrame, logger: logging.Logger) -> None:
    try:
        pivot_dist = pd.crosstab(final_df["AnalysisGroup"], final_df["t5_label"], normalize="index")
        pivot_dist = pivot_dist.reindex([group for group in GROUP_ORDER if group in pivot_dist.index])

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_dist, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Utterance-Level Emotion Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "1_utterance_emotion_distribution_group4.png"), dpi=200)
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
        plt.title("Utterance-Level Average Confidence Scores")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "2_utterance_model_confidence_group4.png"), dpi=200)
        plt.close()

        logger.info("Both utterance-level heatmaps generated.")
    except Exception as exc:
        logger.warning(f"Visualization error: {exc}")


def main() -> None:
    logger = setup_logger()

    try:
        sample_df = load_and_sample(logger)
    except Exception as exc:
        logger.error(f"Setup Error: {exc}")
        raise

    tokenizer, model, device = load_model(logger)

    logger.info("Starting T5 emotion classification (UTTERANCE-LEVEL)...")
    results_data = run_t5_with_confidence(sample_df["UTTE_Text"].tolist(), tokenizer, model, device)
    final_df = pd.concat([sample_df.reset_index(drop=True), pd.DataFrame(results_data)], axis=1)

    final_df.loc[~final_df["t5_label"].isin(VALID_LABELS), "t5_label"] = "INVALID"
    final_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved results to: {OUTPUT_FILE}")

    save_heatmaps(final_df, logger)
    logger.info("=== TASK COMPLETE ===")


if __name__ == "__main__":
    main()
