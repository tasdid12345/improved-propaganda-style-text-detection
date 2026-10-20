# Word + Character N-gram Topic Classifier vs TF-IDF Logistic Regression (20 Newsgroups)

**Motivation.** Flagging propaganda-style online news is a text classification problem, and the paper "Data Analysis Techniques and Detection of Propaganda in Serbian Online Media in 2023" applies data-analysis techniques to this task. Its Serbian corpus is not publicly available, so this experiment uses a public English news-style corpus as a stand-in to test a classifier design choice in a reproducible way.

**Question.** Can a class-balanced word-plus-character n-gram model with calibrated probabilities beat a standard TF-IDF logistic regression at classifying news-style documents by topic, as a stand-in for flagging propaganda-style articles?

**Dataset.** 20 Newsgroups, four categories (alt.atheism, sci.space, comp.graphics, rec.sport.hockey), loaded with scikit-learn's `fetch_20newsgroups` with headers, footers and quotes removed. The official train and test subsets are pooled and re-split once (stratified, 25% test, seed 42) to keep the test share within the 20-30% rule. This is not the paper's dataset.

**Baseline.** TF-IDF word-unigram features with a standard logistic regression, fit on the training split.

**Improved model.** Class-balanced logistic regression on combined word 1-2-gram and character 3-5-gram TF-IDF features. C is chosen by 5-fold cross-validation on the training split only, and the probabilities are calibrated with isotonic regression.

Both models share the same data loading, preprocessing, test split and evaluation code. The main metric is test macro-F1; accuracy and Brier score are secondary.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

## Results

Results are written to `results.json` (and a chart to `figures/`) when the script runs.

## Results (latest run)

On the 4-category 20 Newsgroups test split (940 documents, a stand-in for propaganda flagging), a standard TF-IDF word logistic regression reached macro-F1 0.893, while my class-balanced word-plus-character n-gram model with isotonic calibration reached 0.900, a small gain that may be within single-split noise.

| Metric | Value |
|---|---|
| baseline_f1 | 0.8931 |
| baseline_accuracy | 0.8926 |
| baseline_brier | 0.2526 |
| improved_f1 | 0.8998 |
| improved_accuracy | 0.9 |
| improved_brier | 0.1402 |
| baseline_score | 0.8931 |
| improved_score | 0.8998 |
| improvement | 0.006749 |
| improvement_accuracy | 0.007447 |
| improvement_brier | 0.1124 |
| cv_best_C | 10 |
| cv_macro_f1 | 0.907 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in dataset: topical newsgroup posts, not Serbian propaganda; results do not transfer directly. The official test subset is ~40% of the data, outside the 20-30% test-size rule, so both official subsets were pooled and re-split once (stratified, seed 42). Both models are deterministic given the seed (lbfgs LR, fixed CV folds), so a single run is reported without seed std. Brier is the multiclass Brier score (sum over classes, mean over documents; range 0-2). The test set was used once per model; C was selected only by training-set CV. Single split, so differences of a few points may be within sampling noise.

## Reference

This project is motivated by the following paper:

> D Drašković, M Ogrizović, M Dodović (2024). *Data Analysis Techniques and Detection of Propaganda in Serbian Online Media in 2023*. … Conference on Applied. https://link.springer.com/chapter/10.1007/978-3-031-99201-8_12

Paper author: Prof. Dražen Drašković, University of Belgrade (drazen.draskovic@etf.bg.ac.rs)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
