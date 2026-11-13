"""Run: python3 test_fraudbn.py   (no data file needed)"""
import numpy as np
import fraudbn as fb


def toy(n=40000, seed=0):
    """Class drives A; B copies A with noise (so B links to A); C is pure noise."""
    r = np.random.default_rng(seed)
    y = (r.random(n) < 0.05).astype(np.int64)
    a = np.where(r.random(n) < 0.8, y * 2, r.integers(0, 3, n))
    b = np.where(r.random(n) < 0.7, a, r.integers(0, 3, n))
    c = r.integers(0, 3, n)
    return np.column_stack([a, b, c]), y, [3, 3, 3]


def test_tree_finds_the_real_link():
    X, y, card = toy()
    assert fb.learn_tree(X, y, card, [0, 1, 2]) == {1: 0}, "A-B should be the only link; C is noise"


def test_blanket_drops_noise():
    X, y, card = toy()
    core, _ = fb.blanket(fb.blanket_stability(X, y, card, runs=20, n_jobs=1))
    assert 0 in core and 2 not in core


def test_matches_pgmpy_exact_inference():
    """Our counting posterior must equal pgmpy's variable elimination on the same network."""
    from pgmpy.inference import VariableElimination
    X, y, card = toy()
    names = ["A", "B", "C"]
    m = fb.Model().fit(X, y, card, [0, 1], {1: 0})
    ve = VariableElimination(m.to_pgmpy(names))
    for row in X[:25]:
        q = ve.query(["Class"], evidence={"A": int(row[0]), "B": int(row[1])}, show_progress=False)
        assert abs(q.values[1] - m.predict_proba(row[None])[0]) < 1e-9


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
