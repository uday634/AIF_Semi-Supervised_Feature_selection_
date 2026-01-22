from __future__ import annotations

import math
import random
import numpy as np
import typing
from itertools import count

from sklearn.feature_selection import mutual_info_classif

from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.type_alias import LabelIndex

from capymoa.anomaly._adaptive_isolation_forest import (
    IsolationTree,
    c,
)

__all__ = ["AdaptiveIsolationForestMIFS"]


class AdaptiveIsolationForestMIFS(AnomalyDetector):
    """
    Adaptive Isolation Forest with Semi-Supervised Mutual Information
    Feature Selection (10% label budget)
    """

    def __init__(
        self,
        schema: Schema,
        window_size=256,
        n_trees=100,
        height=None,
        seed: int | None = None,
        m_trees=10,
        weights=0.5,
        label_budget=0.10,
    ):
        super().__init__(schema=schema, random_seed=seed if seed is not None else 1)

        self.n_trees = n_trees
        self.height_limit = height or math.ceil(math.log2(window_size))
        self.window_size = window_size
        self.m_trees = m_trees
        self.weights = weights
        self.label_budget = label_budget

        self.instances: list[Instance] = []
        self.labels: list[int | None] = []

        self._trees: list[IsolationTree] = []
        self.rng = random.Random(self.random_seed)
        self.id_counter = count(start=0)

        self.last_selected_count: int | None = None

    # ------------------------------------------------------------------
    # Mutual Information Feature Selection
    # ------------------------------------------------------------------
    def _feature_selection(self, instances: list[Instance]) -> list[int]:
        num_features = self.schema.get_num_attributes()
        all_features = list(range(num_features))

        # Cold start
        if not self._trees or not instances:
            return all_features

        X = np.array([inst.x[:num_features] for inst in instances])

        # Pseudo labels from current model
        scores = np.array([self.score_instance(inst) for inst in instances])
        pseudo_y = (scores > 0.5).astype(int)

        # Inject 10% real labels
        n = len(self.labels)
        n_known = max(1, int(n * self.label_budget))
        known_indices = random.sample(range(n), n_known)

        y_target = pseudo_y.copy()
        for i in known_indices:
            if self.labels[i] is not None:
                y_target[i] = self.labels[i]

        # Mutual Information
        mi = mutual_info_classif(
            X,
            y_target,
            discrete_features=False,
            random_state=42,
        )

        threshold = np.mean(mi)
        selected = np.where(mi > threshold)[0].tolist()

        # Safety fallback
        if not selected:
            selected = all_features

        self.last_selected_count = len(selected)
        return selected

    # ------------------------------------------------------------------
    # Tree scoring (same as original AIF)
    # ------------------------------------------------------------------
    def _compute_tree_scores(self, trees: list[IsolationTree]) -> list[float]:
        if not trees:
            return []

        tree_sizes = [t.n_nodes for t in trees]
        max_masses = [t.max_mass for t in trees]

        min_size, max_size = min(tree_sizes), max(tree_sizes)
        norm_sizes = (
            [0.0] * len(trees)
            if max_size == min_size
            else [(max_size - s) / (max_size - min_size) for s in tree_sizes]
        )

        min_mass, max_mass = min(max_masses), max(max_masses)
        norm_masses = (
            [0.0] * len(trees)
            if max_mass == min_mass
            else [(m - min_mass) / (max_mass - min_mass) for m in max_masses]
        )

        return [
            self.weights * norm_sizes[i] + (1 - self.weights) * norm_masses[i]
            for i in range(len(trees))
        ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, instance: Instance, label: int | None = None):
        self.instances.append(instance)
        self.labels.append(label)

        if len(self.instances) < self.window_size:
            return

        # ---- FEATURE SELECTION ----
        features = self._feature_selection(self.instances)

        # ---- TREE UPDATE ----
        if not self._trees:
            while len(self._trees) < self.n_trees:
                t = IsolationTree(
                    self.instances,
                    features,
                    self.height_limit,
                    next(self.id_counter),
                    self.rng,
                )
                self._trees.append(t)
        else:
            candidates = [
                IsolationTree(
                    self.instances,
                    features,
                    self.height_limit,
                    next(self.id_counter),
                    self.rng,
                )
                for _ in range(self.m_trees)
            ]

            all_trees = candidates + self._trees
            scores = self._compute_tree_scores(all_trees)

            cand_scores = scores[: self.m_trees]
            tree_scores = scores[self.m_trees :]

            best_cand = cand_scores.index(max(cand_scores))
            worst_tree = tree_scores.index(min(tree_scores))

            if cand_scores[best_cand] > tree_scores[worst_tree]:
                del self._trees[worst_tree]
            else:
                del self._trees[0]

            self._trees.append(candidates[best_cand])

        # Reset window
        self.instances = []
        self.labels = []

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score_instance(self, instance: Instance) -> float:
        if not self._trees:
            return 0.5

        score = sum(t.score_instance(instance) for t in self._trees)
        score /= len(self._trees)
        score /= c(self.window_size)

        return 2 ** (-score)

    def predict(self, instance) -> typing.Optional[LabelIndex]:
        raise NotImplementedError
