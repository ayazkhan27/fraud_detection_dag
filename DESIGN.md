# Design decisions

Goal, unchanged from version 1: a fraud detector that a person can read, with a network
picture from which a stakeholder can see which factors matter.

Each item below is marked **Assumption**, **Evidence** or **Decision**.
All evidence in sections 1 to 5 comes from the training and tuning slices.
The test slice was opened once, after every choice here was fixed.

## Criteria, set before comparing designs

1. Area under the precision-recall curve on the tuning slice.
2. A blanket small enough to read (about ten features or fewer).
3. The blanket should not change when the data are resampled.
4. Every probability table should be backed by real frauds, not synthetic ones.
5. No claim the data cannot support (direction of cause in particular).

## 1. What a Markov blanket can and cannot say

- **Evidence (theory).** The Markov blanket of a variable is the same set of variables in every network that represents the same distribution. Arrow directions differ between such networks; the blanket does not.
- **Decision.** Report the blanket as "the features that carry the signal". Make no statement about which causes which.
- **Assumption.** The arrows are drawn from fraud to features. This is a modelling convention (fraud status generates the transaction's look), not a finding.

## 2. Designs compared on the tuning slice (57 frauds)

| Design | Blanket | Tuning AUPRC | Problem |
|---|---|---|---|
| Version 1 rerun without SMOTE: features point at fraud, one table, greedy BIC | V14, V10 | 0.635 | The table grows 7-fold per parent, so BIC stops at two |
| Naive Bayes, features added while out-of-fold log-loss improves | 2 to 5 features | 0.69 to 0.73 | Admits noise features (V13, V15, V26) after the first two |
| Same, each new feature may lean on one earlier feature | 3 to 12 features | 0.69 to 0.73 | First feature chosen changes with the binning (V14 or V17) |
| **L1 logistic selection with resampling, then tree** | **8 features** | **0.735** | None found; see section 3 |
| XGBoost, all 30 features (reference) | not readable | 0.776 | |
| Logistic regression, all 30 (reference) | not readable | 0.645 | |

- **Not tried.** Unconstrained hill climbing over all 31 variables. It would return arrow directions the data cannot identify, and its blanket cannot be resampled cheaply.
- **Decision.** The fourth design.

## 3. Is the blanket stable?

Share of 100 random half-samples in which each feature was selected, by penalty strength (smaller C is a stronger penalty):

| Feature | C=0.01 | C=0.02 | C=0.05 | C=0.1 | |
|---|---|---|---|---|---|
| V10 | 1.00 | 1.00 | 1.00 | 1.00 | blanket |
| V12 | 1.00 | 1.00 | 1.00 | 1.00 | blanket |
| V14 | 1.00 | 1.00 | 1.00 | 1.00 | blanket |
| V16 | 1.00 | 1.00 | 1.00 | 1.00 | blanket |
| V17 | 1.00 | 1.00 | 1.00 | 1.00 | blanket |
| V21 | 0.98 | 1.00 | 1.00 | 1.00 | blanket |
| V4 | 0.96 | 0.99 | 1.00 | 1.00 | blanket |
| V11 | 0.96 | 0.96 | 0.97 | 0.98 | blanket |
| V19 | 0.00 | 0.61 | 0.99 | 1.00 | borderline |
| Amount | 0.00 | 0.16 | 0.80 | 0.93 | borderline |
| V26 | 0.00 | 0.00 | 0.23 | 0.91 | borderline |
| V3 | 0.52 | 0.39 | 0.28 | 0.44 | out |
| V5 | 0.00 | 0.14 | 0.41 | 0.58 | out |
| V2 | 0.11 | 0.19 | 0.32 | 0.47 | out |
| V7 | 0.20 | 0.21 | 0.18 | 0.31 | out |

The remaining 15 features are selected less often on average than those shown.

- **Evidence.** Eight features are chosen at least 96% of the time at every penalty. Three more appear only when the penalty is weak.
- **Decision.** Blanket rule fixed at "at least 90% at every penalty". The three borderline features are shown on the page as borderline and left out of the model.
- **Evidence (model-free check, test slice).** XGBoost on the 8 blanket features scores 0.786; on all 30 it scores 0.790. Difference -0.004, 95% interval -0.033 to +0.022. A flexible model finds almost nothing outside the blanket.
- **Limit.** Half-samples follow Meinshausen and Bühlmann, but their error bound is not claimed: it is not informative with 30 candidates.

## 4. Bins

- **Decision.** Seven percentile bins (1, 5, 25, 75, 95, 99), cut on the training slice without labels.
- **Evidence.** With eleven bins (adding 0.1, 0.5, 99.5, 99.9) the L1 step selected 15 to 19 features and the set changed with the penalty.
- **Trade-off.** A 1% bin holds about 1,700 transactions, so the strongest single reading is capped near 100 to 1. Finer tails would rank better and be less stable.
- **Decision.** Time of day replaces raw time. Raw seconds from a two-day window have no meaning in a later period.

## 5. Feature links

| On the tuning slice | AUPRC | Log-loss | Mean predicted fraud rate (actual 0.00100) |
|---|---|---|---|
| No links (naive Bayes) | 0.740 | 0.0078 | 0.00171 |
| With the tree | 0.734 | 0.0032 | 0.00104 |

- **Evidence.** The links do not improve ranking. They more than halve log-loss and bring the average predicted rate in line with the actual rate. Without them the model double-counts features that move together and its best threshold sits at 0.9994.
- **Decision.** Keep the tree, because stakeholders read the probabilities. On the test slice the pattern repeated: AUPRC 0.748 with links and 0.756 without, log-loss 0.0041 and 0.0071.
- **Method.** Chow-Liu on mutual information given fraud status, keeping a link only when its likelihood gain exceeds its BIC penalty. This is the exact BIC optimum among one-parent structures.
- **Limit.** The mutual information is dominated by the 99.8% of normal transactions. A tree weighted toward the fraud class might rank better; it was not tried.

## 6. What was removed from version 1, and why

| Removed | Evidence |
|---|---|
| SMOTE-NC | Created about 227,000 synthetic frauds from 394 real ones. BIC then treated 455,000 rows as independent. The fitted base rate was 36%, against a true 0.17% |
| Edge whitelist | Allowed only feature-to-fraud arrows, so the search could never test direction |
| "Tabu search escapes local optima" | pgmpy's loop stops as soon as the best move gains less than 0.0001; it never takes a downhill step |
| Do-calculus step | All three parents of fraud were root nodes, so do(X=x) equalled P(fraud given X=x) |
| Duplicate interaction columns | `V14_V12` and `V12_V14` were both created. Fraud attached to one and the raw features to the other, leaving V12 and V14 with no path to fraud |
| Lean proofs | Every theorem ended in `sorry` |
| Random split | Test rows came from the same hours as training rows |

## 7. Known limits

- 75 frauds in the test slice and 57 in the tuning slice. Differences of a few points between designs in section 2 are within noise.
- Two days of data from 2013.
- The network trails XGBoost by 0.042 AUPRC on the test slice (95% interval 0.006 to 0.087).
- The model was fitted on the first 60% only; it was not refitted on the tuning slice.
