"""End to end: data -> blanket -> network -> test metrics -> docs/index.html.

    python3 run.py                 # expects data/creditcard.csv (Kaggle mlg-ulb/creditcardfraud)
"""
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fraudbn as fb

ROOT = Path(__file__).parent
F = fb.FEATURES
BUDGET = 0.001          # "we can review 1 in 1,000 transactions"
BOOT = 1000


def bin_labels(f, edges, share):
    if f == "Hour":
        e = [0] + [int(x) for x in edges] + [24]
        return [f"{a:02d}:00 to {b:02d}:00" for a, b in zip(e[:-1], e[1:])]
    if len(edges) == len(fb.PCTS):
        return ["lowest 1%", "low 1-5%", "low 5-25%", "middle 50%", "high 75-95%", "high 95-99%", "highest 1%"]
    cum = np.concatenate([[0], np.cumsum(share)]) * 100
    return [f"percentile {a:.0f}-{b:.0f}" for a, b in zip(cum[:-1], cum[1:])]


def at_threshold(y, s, thr):
    hit = s >= thr
    tp, fp, fn = int((hit & (y == 1)).sum()), int((hit & (y == 0)).sum()), int((~hit & (y == 1)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(threshold=float(thr), tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-12))


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    return float(t[np.argmax(2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12))])


def main():
    df = fb.load(ROOT / "data" / "creditcard.csv")
    tr, va, te = fb.split(df)
    ytr, yva, yte = (d["Class"].to_numpy().astype(np.int64) for d in (tr, va, te))
    edges = fb.fit_bins(tr)
    card = [len(edges[f]) + 1 for f in F]
    Xtr, Xva, Xte = (fb.encode(d, edges) for d in (tr, va, te))

    # 1. blanket
    freq = fb.blanket_stability(Xtr, ytr, card)
    core, borderline = fb.blanket(freq)
    print("blanket:", [F[j] for j in core], "| borderline:", [F[j] for j in borderline])

    # 2-3. structure and tables
    parent = fb.learn_tree(Xtr, ytr, card, core)
    bn = fb.Model().fit(Xtr, ytr, card, core, parent)
    from pgmpy.readwrite import BIFWriter
    (ROOT / "docs").mkdir(exist_ok=True)
    BIFWriter(bn.to_pgmpy()).write_bif(str(ROOT / "docs" / "network.bif"))

    # comparison models, fixed settings, never tuned on the test slice
    cols = [F[j] for j in core]
    xkw = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
               random_state=0, n_jobs=8)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[F], ytr)
    xg_all = xgb.XGBClassifier(**xkw).fit(tr[F], ytr)
    xg_bl = xgb.XGBClassifier(**xkw).fit(tr[cols], ytr)
    naive = fb.Model().fit(Xtr, ytr, card, core)
    models = {
        "Bayesian network (8 blanket features)": (bn.predict_proba(Xva), bn.predict_proba(Xte)),
        "Same features, no feature links (naive Bayes)": (naive.predict_proba(Xva), naive.predict_proba(Xte)),
        "XGBoost on the 8 blanket features": (xg_bl.predict_proba(va[cols])[:, 1], xg_bl.predict_proba(te[cols])[:, 1]),
        "XGBoost on all 30 features": (xg_all.predict_proba(va[F])[:, 1], xg_all.predict_proba(te[F])[:, 1]),
        "Logistic regression on all 30 features": (lr.predict_proba(va[F])[:, 1], lr.predict_proba(te[F])[:, 1]),
    }

    # test metrics; thresholds come from the validation slice
    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
    boots = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(BOOT)]
    k = int(round(BUDGET * len(yte)))
    metrics, ap_boot = [], {}
    for name, (sv, st) in models.items():
        ap_boot[name] = np.array([average_precision_score(yte[b], st[b]) for b in boots])
        top = np.argsort(-st, kind="stable")[:k]
        m = at_threshold(yte, st, best_f1_threshold(yva, sv))
        m.update(model=name, ap=float(average_precision_score(yte, st)),
                 ap_lo=float(np.percentile(ap_boot[name], 2.5)), ap_hi=float(np.percentile(ap_boot[name], 97.5)),
                 valid_ap=float(average_precision_score(yva, sv)),
                 budget_alerts=k, budget_caught=int(yte[top].sum()),
                 logloss=float(-np.mean(yte * np.log(np.clip(st, 1e-12, 1)) + (1 - yte) * np.log(np.clip(1 - st, 1e-12, 1)))))
        metrics.append(m)
    names = list(models)
    gap = ap_boot[names[0]] - ap_boot[names[3]]
    gap_bl = ap_boot[names[2]] - ap_boot[names[3]]

    print("\n| Model | AUPRC (95%% interval) | Precision | Recall | F1 | Caught in top %d |" % k)
    print("|---|---|---|---|---|---|")
    for m in metrics:
        print("| %s | %.3f (%.3f to %.3f) | %.3f | %.3f | %.3f | %d of %d |" % (
            m["model"], m["ap"], m["ap_lo"], m["ap_hi"], m["precision"], m["recall"], m["f1"],
            m["budget_caught"], int(yte.sum())))
    print("BN minus XGBoost-all AUPRC: %.3f (%.3f to %.3f)" % (gap.mean(), *np.percentile(gap, [2.5, 97.5])))
    print("XGBoost-blanket minus XGBoost-all AUPRC: %.3f (%.3f to %.3f)" % (gap_bl.mean(), *np.percentile(gap_bl, [2.5, 97.5])))

    # everything the dashboard shows
    st = models[names[0]][1]
    C = bn.contrib(Xte) / np.log(2)                                   # bits
    feats = []
    for a, j in enumerate(core):
        n = np.bincount(ytr * card[j] + Xtr[:, j], minlength=2 * card[j]).reshape(2, card[j])
        share = n.sum(0) / n.sum()
        t = bn.marginal[j]
        feats.append(dict(
            name=F[j], parent=F[parent[j]] if j in parent else None,
            stability=[float(x) for x in freq[:, j]],
            bins=[dict(label=lab, lo=None if b == 0 else float(edges[F[j]][b - 1]),
                       hi=None if b == card[j] - 1 else float(edges[F[j]][b]),
                       rows=int(n[:, b].sum()), frauds=int(n[1, b]), share=float(share[b]),
                       bits=float(np.log2(t[1, b] / t[0, b])))
                  for b, lab in enumerate(bin_labels(F[j], edges[F[j]], share))]))
    links = [dict(a=F[p], b=F[c], bits=float(fb.cond_mi(fb._pair_counts(Xtr, ytr, p, c, card[p], card[c])) / np.log(2)))
             for c, p in parent.items()]

    def example(i, why):
        row = te.iloc[i]
        return dict(why=why, fraud=int(yte[i]), prob=float(st[i]), amount=float(row["Amount"]),
                    hour=int(row["Hour"]), bins=[int(Xte[i, j]) for j in core], bits=[float(x) for x in C[i]])
    thr = metrics[0]["threshold"]
    fr, lg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
    examples = [
        example(int(fr[np.argmax(st[fr])]), "A fraud the model caught"),
        example(int(fr[np.argmin(st[fr])]), "A fraud the model missed"),
        example(int(lg[np.argmax(st[lg])]), "A false alarm"),
        example(int(lg[np.argsort(st[lg])[len(lg) // 2]]), "A typical normal transaction"),
    ]
    grid = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    grid = sorted(set(grid + [round(thr, 4)]))
    curves = {}
    for name in (names[0], names[3], names[4]):
        p, r, _ = precision_recall_curve(yte, models[name][1])
        curves[name] = [[float(x), float(p[r == x].max())] for x in np.unique(r)]   # best precision at each recall level

    hours = lambda d: [float(d["Time"].min() / 3600), float(d["Time"].max() / 3600)]
    data = dict(
        split=dict(train=dict(rows=len(tr), frauds=int(ytr.sum()), hours=hours(tr)),
                   valid=dict(rows=len(va), frauds=int(yva.sum()), hours=hours(va)),
                   test=dict(rows=len(te), frauds=int(yte.sum()), hours=hours(te))),
        prior_bits=bn.prior / np.log(2), base_rate=float(ytr.mean()),
        features=feats, links=links,
        borderline=[dict(name=F[j], stability=[float(x) for x in freq[:, j]]) for j in borderline],
        outside=[dict(name=F[j], stability=[float(x) for x in freq[:, j]])
                 for j in np.argsort(-freq.mean(0)) if j not in core and j not in borderline],
        penalties=fb.PENALTIES, core_rule=fb.CORE,
        metrics=metrics, curves=curves, threshold=thr,
        sweep=[at_threshold(yte, st, t) for t in grid], examples=examples,
        gap=dict(mean=float(gap.mean()), lo=float(np.percentile(gap, 2.5)), hi=float(np.percentile(gap, 97.5))),
        gap_blanket=dict(mean=float(gap_bl.mean()), lo=float(np.percentile(gap_bl, 2.5)), hi=float(np.percentile(gap_bl, 97.5))),
    )
    (ROOT / "docs" / "results.json").write_text(json.dumps(data, indent=1))
    page = (ROOT / "dashboard.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>')
    (ROOT / "docs" / "index.html").write_text(head + page + "</body></html>")
    print("\nwrote docs/index.html, docs/results.json, docs/network.bif")


if __name__ == "__main__":
    sys.exit(main())
