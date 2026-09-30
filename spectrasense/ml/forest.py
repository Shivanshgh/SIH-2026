"""Small JSON-serializable Random Forest implementation for offline use."""

import math
import random


def _gini(counts):
    total = sum(counts)
    return 1.0 - sum((count / total) ** 2 for count in counts if count) if total else 0.0


def _leaf(indices, labels, classes):
    counts = [sum(labels[i] == cls for i in indices) for cls in classes]
    return {"counts": counts}


def _fit_tree(rows, labels, classes, rng, max_depth, min_leaf, max_features):
    feature_count = len(rows[0])

    def grow(indices, depth):
        node = _leaf(indices, labels, classes)
        if depth >= max_depth or len(indices) < min_leaf * 2 or max(node["counts"]) == len(indices):
            return node
        parent_impurity = _gini(node["counts"])
        features = rng.sample(range(feature_count), min(max_features, feature_count))
        best = None
        for feature in features:
            ordered = sorted(set(rows[i][feature] for i in indices))
            if len(ordered) < 2:
                continue
            # Quantile candidates cap training cost while retaining useful splits.
            positions = sorted(set(int((len(ordered) - 1) * q / 16) for q in range(1, 16)))
            for pos in positions:
                threshold = (ordered[pos] + ordered[pos + 1]) / 2.0
                left = [i for i in indices if rows[i][feature] <= threshold]
                right = [i for i in indices if rows[i][feature] > threshold]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                left_counts = [sum(labels[i] == cls for i in left) for cls in classes]
                right_counts = [sum(labels[i] == cls for i in right) for cls in classes]
                gain = parent_impurity - (len(left) * _gini(left_counts) + len(right) * _gini(right_counts)) / len(indices)
                if best is None or gain > best[0]:
                    best = (gain, feature, threshold, left, right)
        if best is None or best[0] <= 1e-10:
            return node
        _, feature, threshold, left, right = best
        return {"feature": feature, "threshold": threshold,
                "left": grow(left, depth + 1), "right": grow(right, depth + 1)}

    return grow(list(range(len(rows))), 0)


class RandomForest:
    def __init__(self, trees=None, classes=None):
        self.trees = trees or []
        self.classes = classes or []

    @classmethod
    def fit(cls, rows, labels, tree_count=48, seed=2026, max_depth=10, min_leaf=2):
        rng = random.Random(seed)
        classes = sorted(set(labels))
        rows = [[float(value) for value in row] for row in rows]
        labels = list(labels)
        max_features = max(1, int(math.sqrt(len(rows[0]))))
        trees = []
        for _ in range(tree_count):
            sample_indices = [rng.randrange(len(rows)) for _ in rows]
            sampled_rows = [rows[i] for i in sample_indices]
            sampled_labels = [labels[i] for i in sample_indices]
            trees.append(_fit_tree(sampled_rows, sampled_labels, classes, rng,
                                   max_depth, min_leaf, max_features))
        return cls(trees, classes)

    def predict_proba_one(self, row):
        totals = [0.0] * len(self.classes)
        for tree in self.trees:
            node = tree
            while "feature" in node:
                node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            counts = node["counts"]
            count_total = sum(counts) or 1
            for index, count in enumerate(counts):
                totals[index] += count / count_total
        total = sum(totals) or 1.0
        return [value / total for value in totals]

    def to_dict(self):
        return {"classes": self.classes, "trees": self.trees}

    @classmethod
    def from_dict(cls, payload):
        return cls(payload["trees"], payload["classes"])
