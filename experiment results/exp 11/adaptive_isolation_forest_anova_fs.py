# adaptive_isolation_forest_anova_fs.py

from __future__ import annotations

import math
import random
import typing
from itertools import count

import numpy as np
from sklearn.feature_selection import f_classif

from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.type_alias import LabelIndex

# ============================================================
# INTERNAL ISOLATION TREE STRUCTURES (copied from AIF)
# ============================================================

class AIFLeaf:
    def __init__(self, X, up, side):
        self.instances = list(X)
        self.up = up
        self.side = side

    def walk(self, instance):
        yield self

    @property
    def mass(self):
        return len(self.instances)

    @property
    def depth(self):
        return 1 + self.up.depth if self.up else 0

    @property
    def n_nodes(self):
        return 1


class AIFBranch:
    ROOT = "root"
    LEFT = "left"
    RIGHT = "right"

    def __init__(self, X, left, right, feature, split_value, up, side):
        self.children = [left, right]
        self.feature = feature
        self.split_value = split_value
        self.instances = list(X)
        self.up = up
        self.side = side

    def walk(self, instance):
        yield self
        yield from self.next(instance).walk(instance)

    def next(self, instance):
        return self.children[0] if instance.x[self.feature] < self.split_value else self.children[1]

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]

    @property
    def mass(self):
        return self.left.mass + self.right.mass

    @property
    def depth(self):
        return 1 + self.up.depth if self.up else 0

    @property
    def n_nodes(self):
        return 1 + self.left.n_nodes + self.right.n_nodes


def make_isolation_tree(X, *, height, rng, attributes, up=None, side=AIFBranch.ROOT):
    if height == 0 or len(X) <= 1:
        return AIFLeaf(X, up, side)

    attrs = attributes.copy()
    while attrs:
        f = rng.choice(attrs)
        values = [inst.x[f] for inst in X]
        mn, mx = min(values), max(values)
        if mn != mx:
            break
        attrs.remove(f)
    else:
        return AIFLeaf(X, up, side)

    split = rng.uniform(mn, mx)

    left = make_isolation_tree(
        [i for i in X if i.x[f] < split],
        height=height - 1,
        rng=rng,
        attributes=attributes,
        up=None,
        side=AIFBranch.LEFT,
    )

    right = make_isolation_tree(
        [i for i in X if i.x[f] >= split],
        height=height - 1,
        rng=rng,
        attributes=attributes,
        up=None,
        side=AIFBranch.RIGHT,
    )

    node = AIFBranch(X, left, right, f, split, up, side)
    left.up = right.up = node
    return node


def H(i):
    return math.log(i) + 0.5772156649 if i > 1 else 0


def c(n):
    return 2 * H(n - 1) - (2 * (n - 1) / n) if n > 1 else 1


class IsolationTree:
    def __init__(self, X, features, height_limit, tree_id, rng):
        self.id = tree_id
        self.root = make_isolation_tree(
            X,
            height=height_limit,
            rng=rng,
            attributes=features,
        )

    def score_instance(self, instance):
        depth = 0
        node = None
        for node in self.root.walk(instance):
            depth += 1
        if node.mass > 1:
            depth += c(node.mass)
        return depth

    @property
    def n_nodes(self):
        return self.root.n_nodes


# ============================================================
# AIF + ANOVA FEATURE SELECTION (10% LABEL BUDGET)
# ============================================================

class AdaptiveIsolationForestWithAnovaFS(AnomalyDetector):

    def __init__(
        self,
        schema: Schema,
        window_size=256,
        n_trees=50,
        label_budget=0.10,
        seed=42,
    ):
        super().__init__(schema=schema, random_seed=seed)
        self.window_size = window_size
        self.n_trees = n_trees
        self.label_budget = label_budget

        self.instances: list[Instance] = []
        self.labels: list[int | None] = []

        self.trees: list[IsolationTree] = []
        self.height_limit = math.ceil(math.log2(window_size))
        self.rng = random.Random(seed)
        self.id_counter = count()

        self.last_selected_count = None

    # --------------------------------------------------------
    # FEATURE SELECTION: ANOVA F-TEST (SEMI-SUPERVISED)
    # --------------------------------------------------------
    def _feature_selection(self):
        num_features = self.schema.get_num_attributes()
        all_features = list(range(num_features))

        X = np.array([inst.x[:num_features] for inst in self.instances])
        scores = np.array([self.score_instance(inst) for inst in self.instances])

        pseudo_y = (scores > 0.5).astype(int)

        n = len(self.instances)
        n_known = max(1, int(n * self.label_budget))
        known_idx = random.sample(range(n), n_known)

        y = pseudo_y.copy()
        for i in known_idx:
            if self.labels[i] is not None:
                y[i] = self.labels[i]

        variances = np.var(X, axis=0)
        valid = np.where(variances > 1e-8)[0]
        if len(valid) == 0:
            return all_features

        f_scores, _ = f_classif(X[:, valid], y)
        f_scores = np.nan_to_num(f_scores)

        threshold = np.mean(f_scores)
        selected = valid[np.where(f_scores > threshold)[0]].tolist()

        self.last_selected_count = len(selected)
        return selected if selected else all_features

    # --------------------------------------------------------
    def train(self, instance: Instance, label: int | None = None):
        self.instances.append(instance)
        self.labels.append(label if random.random() < self.label_budget else None)

        if len(self.instances) < self.window_size:
            return

        features = self._feature_selection()

        while len(self.trees) < self.n_trees:
            self.trees.append(
                IsolationTree(
                    self.instances,
                    features,
                    self.height_limit,
                    next(self.id_counter),
                    self.rng,
                )
            )

        self.instances.clear()
        self.labels.clear()

    def score_instance(self, instance: Instance) -> float:
        if not self.trees:
            return 0.5

        s = sum(t.score_instance(instance) for t in self.trees) / len(self.trees)
        s /= c(self.window_size)
        return 2 ** -s

    def predict(self, instance) -> typing.Optional[LabelIndex]:
        raise NotImplementedError
