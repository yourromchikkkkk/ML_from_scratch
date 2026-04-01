import numpy as np
from collections import Counter 

class Node:
    def __init__(self, feature_idx=None, threshold=None, info_gain=None, left=None, right=None, value=None):
        # Decision Node
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right

        # Leaf Node
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2, impurity_method='entropy'):
        allowed = {"gini", "entropy"}

        if impurity_method not in allowed:
            raise ValueError(
                f"impurity_method must be one of {allowed}, got '{impurity_method}'"
            )

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.impurity_method = impurity_method
        self.root = None

    def build_tree(self, dataset, current_depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            best_split = self.best_split(dataset, n_features)

            if best_split['info_gain'] > 0:
                left_node = self.build_tree(best_split['left_dataset'], current_depth = current_depth + 1)
                right_node = self.build_tree(best_split['right_dataset'], current_depth = current_depth + 1)

                return Node(best_split['feature_idx'], best_split['threshold'], best_split['info_gain'], left_node, right_node)
        
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)
    
    def best_split(self, dataset, n_samples):
        best_split = {'feature_idx': None, 'threshold': None, 'info_gain': -1, 'left_dataset': None, 'right_dataset': None }

        for idx in range(n_samples):
            feature_values = dataset[:, idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_dataset, right_dataset = self.split(dataset, idx, threshold)

                if len(left_dataset) and len(right_dataset):
                    parent_y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]

                    info_gain = self.calc_information_gain(parent_y, left_y, right_y, self.impurity_method)

                    if info_gain > best_split['info_gain']:
                        best_split['info_gain'] = info_gain
                        best_split['feature_idx'] = idx
                        best_split['threshold'] = threshold
                        best_split['left_dataset'] = left_dataset
                        best_split['right_dataset'] = right_dataset
    
        return best_split

    def split(self, dataset, feature_idx, threshold):
        left_dataset = np.array([row for row in dataset if row[feature_idx] <= threshold])
        right_dataset = np.array([row for row in dataset if row[feature_idx] > threshold])

        return left_dataset, right_dataset

    def calc_information_gain(self, parent_labels, left_labels, right_labels, purity_method='entropy'):
        left_w = len(left_labels) / len(parent_labels)
        right_w = len(right_labels) / len(parent_labels)

        info_gain = None
        if purity_method == 'entropy':
            info_gain = self.entropy(parent_labels) - ((left_w * self.entropy(left_labels)) + (right_w * self.entropy(right_labels)))
        elif purity_method == 'gini':
            info_gain = self.gini(parent_labels) - ((left_w * self.gini(left_labels)) + (right_w * self.gini(right_labels)))
        else:
            raise RuntimeError('Such putiry calcualtion method is not defined')
        
        return info_gain
        
    def entropy(self, labels):
        entropy = 0

        class_labels = np.unique(labels)
        for label in class_labels:
            p = len(labels[labels == label]) / len(labels)
            entropy += -p * np.log2(p)
        
        return entropy
    
    def gini(self, labels):
        p_sum = 0

        class_labels = np.unique(labels)
        for label in class_labels:
            p = len(labels[labels == label]) / len(labels)
            p_sum += p ** 2
        
        return 1 - p_sum
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        dataset = np.concatenate([X, y], axis=1)
        self.root = self.build_tree(dataset)

    def predict_class(self, row, node):
        if node.value != None:
            return node.value
        
        feature_val = row[node.feature_idx]
        if feature_val <= node.threshold:
            return self.predict_class(row, node.left)
        else:
            return self.predict_class(row, node.right)

    def predict(self, X):
        X = np.asarray(X)
        return [self.predict_class(row, self.root) for row in X]
    
    def print_tree(self, node=None, depth=0, indent="|   "):
        prefix = indent * depth

        if node is None:
            node = self.root

        if node.value is not None:
            print(f"{prefix}|--- class: {node.value}")
            return

        feature_label = f"Feature {node.feature_idx}"

        print(f"{prefix}|--- {feature_label} <= {node.threshold}")
        self.print_tree(node.left, depth + 1, indent)

        print(f"{prefix}|--- {feature_label} > {node.threshold}")
        self.print_tree(node.right, depth + 1, indent)
        
