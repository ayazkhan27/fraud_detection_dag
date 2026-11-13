"""Readable Bayesian-network fraud classifier.

1. Blanket: which features does Class need? L1-penalised logistic neighbourhood
   selection, repeated on random half-samples (stability selection).
2. Structure: Class -> each blanket feature, plus the best one-parent forest
   among the features (Chow-Liu on I(Xi; Xj | Class), edges kept only if they
   pay their BIC cost). This is a tree-augmented naive Bayes (Friedman et al. 1997).
3. Parameters: smoothed counts. No sampling, no synthetic rows.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from sklearn.linear_model import LogisticRegression

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]
PCTS = [1, 5, 25, 75, 95, 99]      # tail-heavy cut points: fraud lives in the tails
HOUR_EDGES = [4, 8, 12, 16, 20]    # six four-hour blocks
M = 5.0                            # smoothing: each table row is pulled toward its simpler table by M pseudo-rows
PENALTIES = [0.01, 0.02, 0.05, 0.1]  # L1 strengths (sklearn C) the blanket must survive
CORE = 0.9                         # a feature is in the blanket if picked in >= 90% of half-samples at every penalty


def load(path):
    df = pd.read_csv(path).sort_values("Time", kind="stable").reset_index(drop=True)
    df["Hour"] = (df["Time"] % 86400) // 3600
    return df


def split(df, train=0.6, valid=0.2):
    """Time-ordered: fit on the past, make choices on the next slice, test on the last."""
    a, b = int(len(df) * train), int(len(df) * (train + valid))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def fit_bins(train):
    """Percentile bins from the training slice. Labels are not used, so binning cannot leak them."""
    return {f: np.array(HOUR_EDGES, float) if f == "Hour"
            else np.unique(np.percentile(train[f], PCTS)) for f in FEATURES}


def encode(df, edges):
    return np.column_stack([np.digitize(df[f].to_numpy(), edges[f]) for f in FEATURES]).astype(np.int64)


# ---------- 1. blanket ----------

def onehot(X, card):
    off = np.concatenate([[0], np.cumsum(card)[:-1]])
    rows = np.repeat(np.arange(len(X)), X.shape[1])
    return sp.csr_matrix((np.ones(X.size), (rows, (X + off).ravel())), shape=(len(X), int(sum(card))))


def _picked(O, y, card, C, seed):
    """Features with any non-zero bin coefficient in an L1 logistic fit on a random half."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum() // 2, replace=False) for c in (0, 1)])
    coef = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=300).fit(O[idx], y[idx]).coef_[0]
    return np.isin(np.arange(len(card)), np.repeat(np.arange(len(card)), card)[np.abs(coef) > 1e-8])


def blanket_stability(X, y, card, runs=100, n_jobs=-1):
    """Selection frequency per feature, shape [len(PENALTIES), n_features]."""
    from joblib import Parallel, delayed
    O = onehot(X, card)
    return np.array([np.mean(Parallel(n_jobs=n_jobs)(delayed(_picked)(O, y, card, C, s) for s in range(runs)), 0)
                     for C in PENALTIES])


def blanket(freq):
    """(core, borderline): core survives every penalty, borderline only some."""
    order = [int(j) for j in np.argsort(-freq.mean(0), kind="stable")]
    core = [j for j in order if freq[:, j].min() >= CORE]
    edge = [j for j in order if freq[:, j].max() >= CORE and j not in core]
    return core, edge


# ---------- 2. structure ----------

def _pair_counts(X, y, i, j, ki, kj):
    return np.bincount((y * ki + X[:, i]) * kj + X[:, j], minlength=2 * ki * kj).reshape(2, ki, kj).astype(float)


def cond_mi(n):
    """I(Xi; Xj | Class) in nats from counts [2, ki, kj]."""
    p = n / n.sum()
    pc, pic, pjc = p.sum((1, 2), keepdims=True), p.sum(2, keepdims=True), p.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nansum(p * np.log(p * pc / (pic * pjc)))


def learn_tree(X, y, card, S):
    """{child: parent} for the BIC-optimal forest over the blanket features S.
    Which end of a link is 'parent' is arbitrary: reversing a tree edge gives an equivalent model."""
    n, m = len(y), len(S)
    W = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            i, j = S[a], S[b]
            gain = n * cond_mi(_pair_counts(X, y, i, j, card[i], card[j])) \
                - 0.5 * np.log(n) * 2 * (card[i] - 1) * (card[j] - 1)
            W[a, b] = max(gain, 0)
    if not W.any():
        return {}
    T = minimum_spanning_tree(-W).toarray()
    T = ((T + T.T) != 0).astype(float)
    parent, seen = {}, set()
    for root in range(m):
        if root in seen:
            continue
        order, pred = breadth_first_order(T, root, directed=False)
        seen.update(order.tolist())
        for v in order[1:]:
            parent[S[v]] = S[pred[v]]
    return parent


# ---------- 3. parameters and inference ----------

def _cond(counts, prior):
    """Smoothed P(x | row) for a counts table [..., x], pulled toward `prior`."""
    return (counts + M * prior) / (counts.sum(-1, keepdims=True) + M)


def nb_tables(X, y, card):
    """Per-feature P(x | class), shape [2, card], pulled toward P(x)."""
    out = []
    for j, k in enumerate(card):
        n = np.bincount(y * k + X[:, j], minlength=2 * k).reshape(2, k).astype(float)
        out.append(_cond(n, n.sum(0) / n.sum()))
    return out


class Model:
    """Class -> blanket features, each with at most one feature parent."""

    def fit(self, X, y, card, S, parent=None):
        self.S, self.parent, self.card = list(S), dict(parent or {}), list(card)
        p = (y.sum() + 1) / (len(y) + 2)
        self.prior = float(np.log(p / (1 - p)))
        self.marginal = nb_tables(X, y, card)                 # P(x | class), used for the per-feature view
        self.tables = {j: _cond(_pair_counts(X, y, self.parent[j], j, card[self.parent[j]], card[j]),
                                self.marginal[j][:, None, :]) if j in self.parent else self.marginal[j]
                       for j in self.S}
        return self

    def contrib(self, X):
        """[rows, len(S)] evidence for fraud in log-odds. Prior + row sum = posterior log-odds."""
        cols = []
        for j in self.S:
            t = self.tables[j]
            if j in self.parent:
                xp = X[:, self.parent[j]]
                cols.append(np.log(t[1, xp, X[:, j]]) - np.log(t[0, xp, X[:, j]]))
            else:
                cols.append(np.log(t[1, X[:, j]]) - np.log(t[0, X[:, j]]))
        return np.column_stack(cols)

    def logodds(self, X):
        return self.prior + self.contrib(X).sum(1)

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-self.logodds(X)))

    def to_pgmpy(self, names=FEATURES):
        """The same network as a pgmpy object, for standard tools and the BIF file."""
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.models import BayesianNetwork
        bn = BayesianNetwork([("Class", names[j]) for j in self.S]
                             + [(names[p], names[c]) for c, p in self.parent.items()])
        p1 = 1 / (1 + np.exp(-self.prior))
        bn.add_cpds(TabularCPD("Class", 2, [[1 - p1], [p1]]))
        for j in self.S:
            t = self.tables[j]
            if j in self.parent:
                p = self.parent[j]                            # t is [class, parent, x] -> columns ordered (class, parent)
                bn.add_cpds(TabularCPD(names[j], self.card[j], t.reshape(-1, self.card[j]).T,
                                       evidence=["Class", names[p]], evidence_card=[2, self.card[p]]))
            else:
                bn.add_cpds(TabularCPD(names[j], self.card[j], t.T, evidence=["Class"], evidence_card=[2]))
        bn.check_model()
        return bn
