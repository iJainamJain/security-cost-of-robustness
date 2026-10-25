"""FLD ensemble classifier for steganalysis (Kodovsky, Fridrich & Holub, 2011).

The standard classifier for high-dimensional handcrafted steganalysis features.
A single FLD on 8000 DCTR dimensions with only a few thousand training examples
would be hopelessly overdetermined; the ensemble instead trains many FLDs on
random low-dimensional subspaces and votes.

Implemented on scikit-learn's LDA rather than the DDE lab's MATLAB, because
MATLAB's Statistics and Machine Learning Toolbox is not installed on this
machine and sklearn is (measured: 0.30 s per base learner at d_sub=800,
n=4000 — ~2.5 min for 500 learners).

Deviation worth stating in the paper: the reference implementation selects
``d_sub`` and ``L`` automatically by out-of-bag error. Here they are explicit
arguments with the reference defaults, and :func:`search_d_sub` reproduces the
OOB search when it matters.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

__all__ = ["EnsembleFLD", "oob_error", "search_d_sub"]


@dataclasses.dataclass
class EnsembleFLD:
    """Ensemble of FLDs on random feature subspaces.

    Attributes
    ----------
    d_sub:
        Dimensionality of each random subspace.
    n_estimators:
        Number of base learners (``L`` in the paper).
    seed:
        RNG seed; fixed so a run reproduces.
    """

    d_sub: int = 800
    n_estimators: int = 500
    seed: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleFLD":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        n_features = X.shape[1]
        d = min(self.d_sub, n_features)

        rng = np.random.default_rng(self.seed)
        self._subspaces: list[np.ndarray] = []
        self._learners: list[LDA] = []
        self._bags: list[np.ndarray] = []
        n = X.shape[0]

        for _ in range(self.n_estimators):
            cols = rng.choice(n_features, d, replace=False)
            # Bootstrap so out-of-bag error is available, as in the reference.
            bag = rng.choice(n, n, replace=True)
            # shrinkage='auto' (Ledoit-Wolf) stands in for the reference's
            # regularised within-class scatter; without it the covariance is
            # singular whenever d_sub approaches the bag size.
            clf = LDA(solver="lsqr", shrinkage="auto")
            clf.fit(X[np.ix_(bag, cols)], y[bag])
            self._subspaces.append(cols)
            self._learners.append(clf)
            self._bags.append(bag)

        self._n_train = n
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Mean vote margin. Larger means more likely stego (class 1)."""
        X = np.asarray(X, dtype=np.float64)
        votes = np.zeros(X.shape[0], dtype=np.float64)
        for cols, clf in zip(self._subspaces, self._learners):
            votes += clf.predict(X[:, cols]).astype(np.float64)
        return votes / len(self._learners)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.5).astype(int)


def oob_error(model: EnsembleFLD, X: np.ndarray, y: np.ndarray) -> float:
    """Out-of-bag error — the reference's criterion for choosing d_sub."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n = X.shape[0]
    votes = np.zeros(n)
    counts = np.zeros(n)
    for cols, clf, bag in zip(model._subspaces, model._learners, model._bags):
        oob = np.setdiff1d(np.arange(n), bag, assume_unique=False)
        if oob.size == 0:
            continue
        votes[oob] += clf.predict(X[np.ix_(oob, cols)])
        counts[oob] += 1
    seen = counts > 0
    pred = (votes[seen] / counts[seen] >= 0.5).astype(int)
    return float(np.mean(pred != y[seen]))


def search_d_sub(X: np.ndarray, y: np.ndarray,
                 candidates: Sequence[int] = (200, 400, 800, 1600),
                 n_estimators: int = 100, seed: int = 0) -> tuple[int, dict]:
    """Pick d_sub by out-of-bag error, as the reference implementation does.

    Uses fewer base learners than the final model — the OOB curve's *shape* is
    what matters here, not its absolute level.
    """
    scores = {}
    for d in candidates:
        m = EnsembleFLD(d_sub=d, n_estimators=n_estimators, seed=seed).fit(X, y)
        scores[d] = oob_error(m, X, y)
    best = min(scores, key=scores.get)
    return best, scores
