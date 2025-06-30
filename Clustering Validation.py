# Submit this file to Gradescope
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import math

class Solution:

    def confusion_matrix(self, true_labels: List[int], pred_labels: List[int]) -> Dict[Tuple[int, int], int]:
        """Calculate the confusion matrix and return it as a sparse matrix in dictionary form.
        Args:
          true_labels: list of true labels
          pred_labels: list of predicted labels
        Returns:
          A dictionary of (true_label, pred_label): count
        """
        cm = defaultdict(int)
        for t, p in zip(true_labels, pred_labels):
            cm[(t, p)] += 1
        return dict(cm)

    def jaccard(self, true_labels: List[int], pred_labels: List[int]) -> float:
        """Calculate the Jaccard index.
        Args:
          true_labels: list of true cluster labels
          pred_labels: list of predicted cluster labels
        Returns:
          The Jaccard index. Do NOT round this value.
        """
        m11 = m10 = m01 = m00 = 0
        n = len(true_labels)
        for i in range(n):
            for j in range(i + 1, n):
                same_true = true_labels[i] == true_labels[j]
                same_pred = pred_labels[i] == pred_labels[j]
                if same_true and same_pred:
                    m11 += 1
                elif not same_true and same_pred:
                    m10 += 1
                elif same_true and not same_pred:
                    m01 += 1
                else:
                    m00 += 1
        return m11 / (m11 + m10 + m01) if (m11 + m10 + m01) > 0 else 0.0

    def nmi(self, true_labels: List[int], pred_labels: List[int]) -> float:
        n = len(true_labels)
        if n == 0:
            return 1.0

        # Frequency counts
        true_counts = Counter(true_labels)
        pred_counts = Counter(pred_labels)

        # Joint frequency counts
        joint_counts = defaultdict(int)
        for t, p in zip(true_labels, pred_labels):
            joint_counts[(t, p)] += 1

        # Mutual Information
        mi = 0.0
        for (t, p), joint in joint_counts.items():
            p_uv = joint / n
            p_u = true_counts[t] / n
            p_v = pred_counts[p] / n
            mi += p_uv * math.log(p_uv / (p_u * p_v))

        # Entropies
        h_true = -sum((count / n) * math.log(count / n) for count in true_counts.values())
        h_pred = -sum((count / n) * math.log(count / n) for count in pred_counts.values())

        # Geometric mean normalization
        if h_true == 0 or h_pred == 0:
            return 1.0
        return mi / math.sqrt(h_true * h_pred)