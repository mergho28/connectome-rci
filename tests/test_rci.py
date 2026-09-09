"""RCI fit -> score behaviour on controlled synthetic route maps."""
import numpy as np

from connectome_rci.rci import fit_weights, load_weights, save_weights, score


def _cn_subject():
    # carries the "CN-typical" route on both pairs
    return {(0, 1): [((0, 1), 1.0)], (0, 2): [((0, 2), 1.0)]}


def _disease_subject():
    # carries a rerouted "disease-typical" route
    return {(0, 1): [((0, 3, 1), 1.0)], (0, 2): [((0, 3, 2), 1.0)]}


def test_fit_score_separates_groups():
    ref = [_cn_subject() for _ in range(10)] + [_disease_subject() for _ in range(10)]
    labels = [0] * 10 + [1] * 10
    W = fit_weights(ref, labels)
    assert W["n_cn"] == 10 and W["n_dis"] == 10
    assert set(W["pairs"]) == {(0, 1), (0, 2)}

    cn_rci = score(W, _cn_subject())[1]
    dis_rci = score(W, _disease_subject())[1]
    # A CN-like subject should score higher (more CN-like) than a disease-like one.
    assert cn_rci > dis_rci


def test_score_returns_vector_and_mean():
    ref = [_cn_subject(), _disease_subject()]
    W = fit_weights(ref, [0, 1])
    y, m = score(W, _cn_subject())
    assert len(y) == len(W["pairs"])
    assert np.isclose(m, float(np.mean(y)))


def test_fit_requires_both_classes():
    try:
        fit_weights([_cn_subject(), _cn_subject()], [0, 0])
    except ValueError:
        return
    raise AssertionError("expected ValueError when only one class is present")


def test_weights_roundtrip(tmp_path):
    W = fit_weights([_cn_subject(), _disease_subject()], [0, 1])
    p = tmp_path / "w.pkl"
    save_weights(W, str(p))
    W2 = load_weights(str(p))
    assert W2["pairs"] == W["pairs"]
    assert np.allclose(W2["baseline"], W["baseline"])
