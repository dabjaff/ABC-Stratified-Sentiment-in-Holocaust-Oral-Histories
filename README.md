# ABC-Stratified Sentiment and Agreement Pipeline

This repository contains cleaned Python scripts for running a sentiment-analysis and agreement workflow on Holocaust oral histories.

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


## Project context

These scripts support a workflow built around multi-model sentiment triangulation, an agreement-based ABC taxonomy, and descriptive T5 emotion profiling for Holocaust oral histories.

## Licence
This work is publicly available under the CC BY-NC-SA 4.0 license.

## Citation

If you use this repository, please cite the associated paper:

Daban Q. Jaff. 2026. *ABC-Stratified Sentiment in Holocaust Oral Histories*. Presented at the Second Workshop on Holocaust Testimonies as Language Resources (HTRes 2026), co-located with LREC 2026, Palma de Mallorca, Spain, 11 May 2026.
