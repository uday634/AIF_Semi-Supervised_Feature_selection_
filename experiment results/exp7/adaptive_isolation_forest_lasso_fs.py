# adaptive_isolation_forest_lasso_fs.py
"""
Adaptive Isolation Forest with Semi-Supervised Lasso Feature Selection
Uses only 10% label budget for feature selection guidance
"""

from __future__ import annotations

import math
import random
import typing
from itertools import count

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.type_alias import LabelIndex

# =============================================================================
# INTERNAL AIF CLASSES (UNCHANGED - must be copied from original CapyMOA)
# =============================================================================

class AIFLeaf:
    def __init__(self, X: list[Instance], up, side):
        self.instances = list(X)
        self.up = up
        self.side = side

    def walk(self, instance: Instance):
        yield self

    @property
    def n_nodes(self):
        return 1

    @property
    def mass(self):
        return len(self.instances)

    @property
    def depth(self):
        return 1 + self.up.depth if self.up is not None else 0


class AIFBranch:
    ROOT = "root"
    LEFT = "left"
    RIGHT = "right"

    def __init__(self, X: list[Instance], left, right, feature, split_value, up, side):
        self.children = [left, right]
        self.feature = feature
        self.split_value = split_value
        self.up = up
        self.side = side
        self.instances = list(X)

    @property
    def repr_split(self):
        return f"{self.feature} < {self.split_value:.5f}"

    def walk(self, instance: Instance):
        yield self
        yield from self.next(instance).walk(instance)

    @property
    def n_nodes(self):
        return 1 + sum(child.n_nodes for child in self.children)

    @property
    def left(self):
        return self.children[0]

    @left.setter
    def left(self, value):
        self.children[0] = value

    @property
    def right(self):
        return self.children[1]

    @right.setter
    def right(self, value):
        self.children[1] = value

    @property
    def mass(self):
        return self.left.mass + self.right.mass

    def next(self, instance: Instance):
        value = instance.x[self.feature]
        if value < self.split_value:
            return self.left
        return self.right

    def __repr__(self):
        return str(f"{self.repr_split}@{self.depth}")

    @property
    def depth(self):
        return 1 + self.up.depth if self.up is not None else 0


def make_isolation_tree(
    X: list[Instance],
    *,
    height,
    rng: random.Random,
    attributes,
    up=None,
    side=AIFBranch.ROOT,
):
    _attributes = attributes.copy()

    if height == 0 or len(X) <= 1:
        return AIFLeaf(X, up=up, side=side)

    while _attributes:
        on = rng.choice(_attributes)
        vals = [inst.x[on] for inst in X]
        a, b = min(vals), max(vals)
        if a != b:
            break
        _attributes.remove(on)
    else:
        return AIFLeaf(X, up=up, side=side)

    at = rng.uniform(a, b)

    left = make_isolation_tree(
        [inst for inst in X if inst.x[on] < at],
        height=height-1,
        rng=rng,
        attributes=attributes,
        up=None,
        side=AIFBranch.LEFT
    )

    right = make_isolation_tree(
        [inst for inst in X if inst.x[on] >= at],
        height=height-1,
        rng=rng,
        attributes=attributes,
        up=None,
        side=AIFBranch.RIGHT
    )

    branch = AIFBranch(X, left, right, on, at, up, side)
    left.up = right.up = branch
    return branch


def H(i):
    return math.log(i) + 0.5772156649 if i > 1 else 0


def c(n):
    return 2 * H(n - 1) - (2 * (n - 1) / n) if n > 1 else 1


class IsolationTree:
    def __init__(
        self,
        X: list[Instance],
        features,
        height_limit: int,
        tree_id: int,
        rng: random.Random,
    ):
        self.id = tree_id
        self.features = features
        self.height_limit = height_limit
        self._root = make_isolation_tree(
            X,
            height=height_limit,
            rng=rng,
            attributes=features,
            up=None,
            side=AIFBranch.ROOT,
        )

    def score_instance(self, instance: Instance) -> float:
        score = 0.0
        for node in self._root.walk(instance):
            score += 1

        if node.mass > 1:
            score += c(node.mass)

        return score

    def _get_all_leaves(self, node=None):
        if node is None:
            node = self._root

        if isinstance(node, AIFLeaf):
            return [node]
        return (
            self._get_all_leaves(node.left) +
            self._get_all_leaves(node.right)
        )

    @property
    def max_mass(self):
        leaves = self._get_all_leaves()
        return max((leaf.mass for leaf in leaves), default=0)

    @property
    def n_nodes(self):
        return self._root.n_nodes


# =============================================================================
# SEMI-SUPERVISED VERSION: Lasso with 10% label budget
# =============================================================================

class AdaptiveIsolationForestWithLassoFS(AnomalyDetector):
    """Adaptive Isolation Forest with Semi-Supervised Lasso Feature Selection
    Uses only a 10% label budget for guiding feature selection"""

    def __init__(
        self,
        schema: Schema,
        window_size=256,
        n_trees=50,
        height=None,
        seed: int | None = None,
        m_trees=10,
        weights=0.5,
        lasso_alpha=0.01,
        label_budget=0.10,  # 10% of window instances have known labels
        supervision_weight=0.7,  # 70% real label + 30% score when label known
    ):
        super().__init__(schema=schema, random_seed=seed if seed is not None else 42)
        self.n_trees = n_trees
        self._trees: list[IsolationTree] = []
        self.height_limit = height or math.ceil(math.log2(window_size))
        self.window_size = window_size
        self.instances: list[Instance] = []
        self.labels: list[int | None] = []  # Store labels (None = unlabeled)
        self.rng = random.Random(self.random_seed)
        self.id_counter = count(start=0)
        self.m_trees = m_trees
        self.weights = weights
        self.lasso_alpha = lasso_alpha
        self.label_budget = label_budget
        self.supervision_weight = supervision_weight
        self.last_selected_count = None

    def _feature_selection(self) -> list[int]:
        """Semi-supervised Lasso: use 10% real labels + scores for target"""
        num_features = self.schema.get_num_attributes()
        all_features = list(range(num_features))

        if not self._trees or not self.instances:
            return all_features

        # Features only (exclude appended label)
        X = np.array([inst.x[:num_features] for inst in self.instances])
        scores = np.array([self.score_instance(inst) for inst in self.instances])

        # Simulate label budget: randomly select 10% instances as "known"
        n = len(self.labels)
        n_known = max(1, int(n * self.label_budget))
        known_indices = random.sample(range(n), n_known)

        # Hybrid target: real labels where known, else scores
        hybrid_target = scores.copy()
        for i in known_indices:
            if self.labels[i] is not None:
                hybrid_target[i] = (
                    self.supervision_weight * self.labels[i] +
                    (1 - self.supervision_weight) * scores[i]
                )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lasso = Lasso(alpha=self.lasso_alpha, max_iter=2000, random_state=42)
        lasso.fit(X_scaled, hybrid_target)

        selected = np.where(np.abs(lasso.coef_) > 1e-5)[0].tolist()
        self.last_selected_count = len(selected)

        return selected if selected else all_features

    def _compute_tree_scores(self, trees: list[IsolationTree]) -> list[float]:
        if not trees:
            return []

        tree_sizes = [t.n_nodes for t in trees]
        max_masses = [t.max_mass for t in trees]

        min_s, max_s = min(tree_sizes), max(tree_sizes)
        norm_sizes = [(max_s - s) / (max_s - min_s) if max_s != min_s else 0.0
                      for s in tree_sizes]

        min_m, max_m = min(max_masses), max(max_masses)
        norm_masses = [(m - min_m) / (max_m - min_m) if max_m != min_m else 0.0
                       for m in max_masses]

        return [self.weights * ns + (1 - self.weights) * nm
                for ns, nm in zip(norm_sizes, norm_masses)]

    def train(self, instance: Instance, label: int | None = None):
        """Train with optional label (simulated budget)"""
        self.instances.append(instance)
        # Simulate label budget: only keep label with probability = label_budget
        self.labels.append(label if random.random() < self.label_budget else None)

        if len(self.instances) == self.window_size:
            selected_features = self._feature_selection()

            if not self._trees:
                while len(self._trees) < self.n_trees:
                    t = IsolationTree(
                        self.instances,
                        selected_features,
                        self.height_limit,
                        next(self.id_counter),
                        self.rng,
                    )
                    self._trees.append(t)
            else:
                candidates = [
                    IsolationTree(
                        self.instances,
                        selected_features,
                        self.height_limit,
                        next(self.id_counter),
                        self.rng,
                    )
                    for _ in range(self.m_trees)
                ]

                all_trees = candidates + self._trees
                all_scores = self._compute_tree_scores(all_trees)

                best_candidate_idx = np.argmax(all_scores[:self.m_trees])
                worst_existing_idx = np.argmin(all_scores[self.m_trees:]) + self.m_trees

                if all_scores[best_candidate_idx] > all_scores[worst_existing_idx]:
                    del self._trees[worst_existing_idx - self.m_trees]
                else:
                    del self._trees[0]

                self._trees.append(candidates[best_candidate_idx])

            self.instances = []
            self.labels = []

    def score_instance(self, instance: Instance) -> float:
        if not self._trees:
            return 0.5

        score = sum(t.score_instance(instance) for t in self._trees) / len(self._trees)
        score /= c(self.window_size)
        return 2 ** -score

    def predict(self, instance) -> typing.Optional[LabelIndex]:
        raise NotImplementedError(
            "Use score_instance() for anomaly detection scores."
        )