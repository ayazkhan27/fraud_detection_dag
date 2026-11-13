# Fraud Blanket: a card-fraud detector you can read

A Bayesian network for credit card fraud, built as an alternative to black-box models.
It answers three questions a fraud team asks:

1. **Which features carry the signal?** The Markov blanket of fraud: 8 of 30 features.
2. **Why did this transaction get this score?** Every score is a base rate plus one piece of evidence per feature.
3. **How good is it?** Precision and recall on transactions later in time than anything it was fitted on.

Open [`docs/index.html`](docs/index.html) in a browser for the interactive view.

## Results

Data: the [ULB / Worldline credit card dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud),
284,807 transactions over two days in September 2013, 492 frauds (0.172%).
Fitted on the first 60% in time, tuned on the next 20%, tested once on the last 20% (56,962 transactions, 75 frauds).

| Model | AUPRC (95% interval) | Precision | Recall | F1 | Caught in top 57 |
|---|---|---|---|---|---|
| **Bayesian network (8 blanket features)** | **0.748** (0.654 to 0.833) | 0.831 | 0.720 | 0.771 | 49 of 75 |
| Same features, no feature links (naive Bayes) | 0.756 (0.662 to 0.847) | 0.850 | 0.680 | 0.756 | 50 of 75 |
| XGBoost on the 8 blanket features | 0.786 (0.699 to 0.868) | 0.931 | 0.720 | 0.812 | 54 of 75 |
| XGBoost on all 30 features | 0.790 (0.700 to 0.871) | 0.855 | 0.707 | 0.774 | 52 of 75 |
| Logistic regression on all 30 features | 0.692 (0.591 to 0.793) | 0.735 | 0.667 | 0.699 | 43 of 75 |

- AUPRC is the area under the precision-recall curve. Intervals are from 1,000 bootstrap resamples of the test slice.
- Precision, recall and F1 use each model's own threshold, chosen on the tuning slice.
- "Top 57" is what each model catches when only 1 in 1,000 transactions can be reviewed.

What the numbers say:

- **The blanket holds nearly all the usable signal.** XGBoost on the 8 blanket features is within 0.004 AUPRC of XGBoost on all 30 (95% interval -0.033 to +0.022).
- **Reading the model costs some accuracy.** The Bayesian network trails XGBoost by 0.042 AUPRC (95% interval 0.006 to 0.087).
- **The test is small.** With 75 test frauds, every interval is wide.

## The network

- **Blanket:** V4, V10, V11, V12, V14, V16, V17, V21.
- **Borderline (left out):** V19, Amount, V26. They are selected at weak penalties only.
- **Feature links:** V10-V14, V10-V21, V10-V4, V14-V12, V14-V17, V14-V11, V16-V17.
- **Standard format:** [`docs/network.bif`](docs/network.bif) opens in pgmpy, bnlearn, GeNIe and similar tools.

## How it works

| Step | Method | Why |
|---|---|---|
| Split | By time: 60 / 20 / 20 | A detector is used on the future, so it is tested on the future |
| Bins | Percentiles 1, 5, 25, 75, 95, 99 of the training slice | Fraud sits in the tails; labels are not used, so bins cannot leak them |
| Blanket | L1-penalised logistic regression on 100 random half-samples, at four penalty strengths. Keep features chosen at least 90% of the time at every strength | Gives the features fraud depends on given all the others, with a measure of how stable that choice is |
| Structure | Fraud points to each blanket feature. Among features, the best one-parent forest (Chow-Liu), keeping a link only if it pays its BIC cost | Exact optimum for this family, so there is no greedy graph search |
| Tables | Counts with smoothing toward the simpler table | Rare combinations do not get zero probability |
| Scoring | Bayes' rule: base rate plus each feature's log-likelihood ratio | The explanation is the calculation, not an approximation of it |

No synthetic rows are created. The model is fitted at the true 0.2% fraud rate.

## What it cannot tell you

- **It finds signals, not causes.** A Markov blanket is the same set whichever way the arrows are drawn. These data cannot settle direction, and V1 to V28 are anonymised blends of the bank's original fields.
- **Two days of data.** No weekend, holiday or new fraud scheme has been seen.
- **Feature-link direction is arbitrary.** Reversing a tree link gives an equivalent model.

## Run it

```bash
pip install -r requirements.txt
# put creditcard.csv from Kaggle in data/
python3 test_fraudbn.py     # three self-checks, no data needed
python3 run.py              # about 2 minutes; writes docs/
```

| File | What it is |
|---|---|
| `fraudbn.py` | Bins, blanket selection, tree, tables, scoring (about 170 lines) |
| `run.py` | Runs everything, evaluates, writes `docs/` |
| `dashboard.html` | Page template; `run.py` fills in the data |
| `test_fraudbn.py` | Checks the tree, the blanket, and agreement with pgmpy's exact inference |
| `DESIGN.md` | Each design decision, the alternatives tried, and the evidence |

## What changed from version 1

Version 1 (February 2025) is in the git history. `DESIGN.md` gives the reason for each change. In short:

| Version 1 | Version 2 | Reason |
|---|---|---|
| Random split | Split by time | Random splits test on the same hours the model trained on |
| SMOTE-NC to 50% fraud | No resampling | Half the training rows were synthetic copies of 394 frauds, which overstated the evidence for every edge |
| Whitelisted edges, greedy hill climbing | Stability-selected blanket, exact tree | The whitelist fixed the answer in advance; the search stopped at the first local optimum |
| One 625-cell fraud table | Tables of at most 98 cells | Each cell is backed by more real frauds |
| "Causal do-calculus engine" | Removed | Every intervened node was a root, so each do-query equalled plain conditioning |
| Lean file of unproved theorems | Removed | Every theorem ended in `sorry` |

## References

- Friedman, Geiger, Goldszmidt (1997). Bayesian network classifiers. *Machine Learning* 29.
- Chow, Liu (1968). Approximating discrete probability distributions with dependence trees. *IEEE Trans. Information Theory* 14.
- Meinshausen, Bühlmann (2010). Stability selection. *JRSS B* 72.
- Ravikumar, Wainwright, Lafferty (2010). High-dimensional Ising model selection using L1-regularized logistic regression. *Annals of Statistics* 38.
- Dal Pozzolo, Caelen, Johnson, Bontempi (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE SSCI*. (Dataset source.)
