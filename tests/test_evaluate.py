import numpy as np

from src.evaluate import apply_threshold, tune_threshold
from src.train import compute_class_weights


def test_apply_threshold_argmax():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    assert apply_threshold(p, None).tolist() == [0, 2]


def test_apply_threshold_prefers_neutral():
    p = np.array([[0.5, 0.25, 0.25], [0.1, 0.2, 0.7]])
    assert apply_threshold(p, 0.24).tolist() == [1, 2]


def test_tune_threshold_never_worse_than_argmax():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=200)
    y = rng.integers(0, 3, size=200)
    from sklearn.metrics import f1_score
    t, f1 = tune_threshold(p, y)
    assert f1 >= f1_score(y, p.argmax(1), average="macro") - 1e-12


def test_class_weights_from_data():
    w = compute_class_weights([0, 0, 0, 1, 2, 2], 3)
    assert w[1] > w[2] > w[0]
