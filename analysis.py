"""Topic classification on 20 Newsgroups (4 categories) as a stand-in for propaganda detection.
Baseline: TF-IDF word-unigram logistic regression.
Improved: class-balanced LR on word 1-2gram + char_wb 3-5gram TF-IDF, C chosen by 5-fold CV on train, isotonic calibration.
"""
import json
import os
import socket
import sys
import time
from urllib.error import HTTPError, URLError

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline, make_union

SEED = 42
np.random.seed(SEED)
socket.setdefaulttimeout(60)  # 60 s timeout for the dataset download (urllib)
CATS = ["alt.atheism", "sci.space", "comp.graphics", "rec.sport.hockey"]
SOURCE = "https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset"


def load_data():
    """Fetch both official subsets (headers/footers/quotes removed); retry on HTTP 429/5xx and network errors."""
    for attempt in range(5):
        try:
            kw = dict(categories=CATS, remove=("headers", "footers", "quotes"), random_state=SEED)
            tr = fetch_20newsgroups(subset="train", **kw)
            te = fetch_20newsgroups(subset="test", **kw)
            return list(tr.data) + list(te.data), np.concatenate([tr.target, te.target])
        except (HTTPError, URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code is not None and code != 429 and code < 500:
                raise  # permanent client error: do not retry
            wait = 5 * 2 ** attempt
            print(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    sys.exit("ERROR: could not download 20 Newsgroups")


def evaluate(name, model, Xtr, ytr, Xte, yte):
    """Shared evaluation for both models: fit on train, score macro-F1, accuracy and multiclass Brier on test."""
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)
    onehot = np.eye(len(CATS))[yte]
    brier = np.mean(np.sum((proba - onehot) ** 2, axis=1))  # multiclass Brier, range [0, 2]
    return {f"{name}_f1": float(f1_score(yte, pred, average="macro")),
            f"{name}_accuracy": float(accuracy_score(yte, pred)),
            f"{name}_brier": float(brier)}


def main():
    t0 = time.time()
    X, y = load_data()
    print(f"loaded {len(X)} documents from {len(CATS)} categories")
    # Pool the official subsets and re-split once (stratified, 25% test) to keep the test share within 20-30%
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    print(f"train={len(Xtr)} test={len(Xte)}")

    # Baseline: standard TF-IDF word unigrams + logistic regression
    baseline = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2),
                             LogisticRegression(max_iter=2000, random_state=SEED))

    # Improved: word 1-2gram + char_wb 3-5gram TF-IDF, class-balanced LR
    feats = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2))
    lr = make_pipeline(feats, LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))
    # Choose C by 5-fold CV on the training split only (macro-F1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(lr, {"logisticregression__C": [0.3, 1, 3, 10]},
                          cv=cv, scoring="f1_macro", n_jobs=-1)
    search.fit(Xtr, ytr)
    best_C = float(search.best_params_["logisticregression__C"])
    print(f"CV-selected C={best_C} (CV macro-F1={search.best_score_:.4f})")
    # Isotonic calibration; internal 5-fold uses only the training split
    improved = CalibratedClassifierCV(search.best_estimator_, method="isotonic", cv=5)

    res_b = evaluate("baseline", baseline, Xtr, ytr, Xte, yte)
    print(f"baseline macro-F1={res_b['baseline_f1']:.4f}")
    res_i = evaluate("improved", improved, Xtr, ytr, Xte, yte)
    print(f"improved macro-F1={res_i['improved_f1']:.4f}")

    m = {**res_b, **res_i}
    m["baseline_score"] = m["baseline_f1"]
    m["improved_score"] = m["improved_f1"]
    m["improvement"] = m["improved_f1"] - m["baseline_f1"]  # higher macro-F1 is better
    m["improvement_accuracy"] = m["improved_accuracy"] - m["baseline_accuracy"]
    m["improvement_brier"] = m["baseline_brier"] - m["improved_brier"]  # lower Brier is better
    m["cv_best_C"] = best_C
    m["cv_macro_f1"] = float(search.best_score_)

    # Sanity check: refuse to write meaningless results
    vals = [v for v in m.values() if isinstance(v, float)]
    if len(set(yte.tolist())) != len(CATS) or not np.all(np.isfinite(vals)):
        sys.exit("ERROR: metrics are not meaningful")

    results = {
        "question": "Can a class-balanced word-plus-character n-gram model with calibrated probabilities "
                    "beat a standard TF-IDF logistic regression at classifying news-style documents by topic, "
                    "as a stand-in for flagging propaganda-style articles?",
        "method": "Pool official 20 Newsgroups train/test subsets (4 categories, headers/footers/quotes removed), "
                  "stratified 75/25 split with seed 42; compare TF-IDF word-unigram LR (baseline) with class-balanced "
                  "LR on word 1-2gram + char_wb 3-5gram TF-IDF, C tuned by 5-fold CV on train, isotonic calibration.",
        "dataset": {
            "name": "20 Newsgroups (4 categories: alt.atheism, sci.space, comp.graphics, rec.sport.hockey)",
            "source_url": SOURCE,
            "n_rows": len(X),
            "n_train": len(Xtr),
            "n_test": len(Xte),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 (test)",
        "higher_is_better": True,
        "baseline": "TF-IDF word unigrams (min_df=2, sublinear tf) + logistic regression (default C=1), fit on train.",
        "improved": "Word 1-2gram + char_wb 3-5gram TF-IDF, class_weight='balanced' logistic regression, "
                    "C selected from {0.3,1,3,10} by 5-fold CV on train only, isotonic probability calibration.",
        "metrics": m,
        "notes": "Stand-in dataset: topical newsgroup posts, not Serbian propaganda; results do not transfer directly. "
                 "The official test subset is ~40% of the data, outside the 20-30% test-size rule, so both official "
                 "subsets were pooled and re-split once (stratified, seed 42). Both models are deterministic given "
                 "the seed (lbfgs LR, fixed CV folds), so a single run is reported without seed std. "
                 "Brier is the multiclass Brier score (sum over classes, mean over documents; range 0-2). "
                 "The test set was used once per model; C was selected only by training-set CV. "
                 "Single split, so differences of a few points may be within sampling noise.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Optional chart: baseline vs improved on test
    os.makedirs("figures", exist_ok=True)
    names = ["macro-F1", "accuracy"]
    b = [m["baseline_f1"], m["baseline_accuracy"]]
    i = [m["improved_f1"], m["improved_accuracy"]]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, b, 0.4, label="baseline")
    ax.bar(x + 0.2, i, 0.4, label="improved")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title("Test-set scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/baseline_vs_improved.png", dpi=120)
    plt.close(fig)
    print(f"done in {time.time() - t0:.1f}s; wrote results.json")


if __name__ == "__main__":
    main()
