# ABC-Stratified Sentiment and Agreement Pipeline

This repository contains cleaned Python scripts for running a sentiment-analysis and agreement workflow on Holocaust oral histories. The scripts support a pipeline that combines three off-the-shelf sentiment classifiers, agreement-based ABC stratification, inter-model agreement diagnostics, and T5-based descriptive emotion profiling.

## What this repository does

The workflow is designed to:

- parse source corpus files into analysis-ready text files
- run three pretrained sentiment models on the same material
- produce sentence-level sentiment outputs and aggregated utterance-level sentiment outputs
- merge model outputs into sentence- and utterance-level summary tables
- assign agreement-based ABC categories
- compute pairwise agreement, Cohen's kappa, Fleiss' kappa, and confusion matrices
- run a T5 emotion model on stratified samples for descriptive comparison across agreement groups

## Repository contents

- `CORHOH_Parser.py`  
  Parses the source corpus and produces `answers.csv`-style text output.

- `NLPTown_cleaned.py`  
  Runs `nlptown/bert-base-multilingual-uncased-sentiment` on the corpus, writes sentence-level predictions, and computes an aggregated utterance-level sentiment.

- `CardNLP_cleaned.py`  
  Runs `cardiffnlp/twitter-roberta-base-sentiment` on the corpus, writes sentence-level predictions, and computes an aggregated utterance-level sentiment.

- `Siebert_cleaned.py`  
  Runs `siebert/sentiment-roberta-large-english` on the corpus, writes sentence-level predictions, and computes an aggregated utterance-level sentiment.

- `Parsing_txt_SA_light_cleaned.py`  
  Parses the model output logs/text files, aligns sentence and utterance results, builds summary tables, and assigns agreement-based categories.

- `Kappa_light_cleaned.py`  
  Computes pairwise percent agreement, Cohen's kappa, Fleiss' kappa, and confusion-matrix outputs for sentence- and utterance-level results.

- `T5_SEN_light_cleaned.py`  
  Runs `mrm8488/t5-base-finetuned-emotion` on sentence-level samples from the ABC groups and creates summary outputs/heatmaps.

- `T5_UTT_light_cleaned.py`  
  Runs `mrm8488/t5-base-finetuned-emotion` on utterance-level samples from the ABC groups and creates summary outputs/heatmaps.

## Suggested workflow

1. Parse the corpus into an `answers.csv`-style file.
2. Run the three sentiment scripts:
   - `NLPTown_cleaned.py`
   - `CardNLP_cleaned.py`
   - `Siebert_cleaned.py`
3. Merge and structure the outputs with `Parsing_txt_SA_light_cleaned.py`.
4. Compute agreement statistics with `Kappa_light_cleaned.py`.
5. Run the T5 descriptive emotion scripts:
   - `T5_SEN_light_cleaned.py`
   - `T5_UTT_light_cleaned.py`

## Main models used

- `nlptown/bert-base-multilingual-uncased-sentiment`
- `cardiffnlp/twitter-roberta-base-sentiment`
- `siebert/sentiment-roberta-large-english`
- `mrm8488/t5-base-finetuned-emotion`

## Typical Python dependencies

The exact environment may vary slightly by script, but the workflow typically uses:

- Python 3.10+
- `pandas`
- `numpy`
- `nltk`
- `torch`
- `transformers`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`

Example install command:

```bash
pip install pandas numpy nltk torch transformers scikit-learn matplotlib seaborn tqdm
```

If NLTK sentence tokenization is not already installed, the scripts may also need:

```bash
python -c "import nltk; nltk.download('punkt')"
```

## Important notes before running

- Several scripts still contain local Windows paths or placeholder paths. Update all input and output paths before running.
- The scripts are workflow-oriented research scripts, not a packaged library.
- Some scripts expect specific column names such as `GlobalBlockID`, `Text`, `UTTE_Text`, `Sentence_ABC_Category`, `Utterance_ABC_Category`, and related sentiment columns.
- Output files are generated script-by-script and are intended to feed the next stage of the workflow.

## Project context

These scripts support a workflow built around multi-model sentiment triangulation, an agreement-based ABC taxonomy, and descriptive T5 emotion profiling for Holocaust oral histories.

## Citation

If you use this workflow, please cite the accompanying paper:

**Daban Q. Jaff. _From Consensus to Split Decisions: ABC-Stratified Sentiment in Holocaust Oral Histories_.**

You may also wish to cite the underlying corpus separately where relevant.
