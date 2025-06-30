# Clustering Validation

This project implements clustering evaluation metrics from scratch using pure Python, without relying on external libraries like NumPy or scikit-learn. It is designed to assess the quality of clustering results against ground truth labels, and is fully compatible with grading environments like Gradescope.

---

## ✅ Implemented Metrics

- **Confusion Matrix**  
  Returns a sparse dictionary of the form `{(true_label, predicted_label): count}`, summarizing co-occurrences between true and predicted labels.

- **Jaccard Index**  
  Computes the proportion of point pairs that are clustered together in both the true and predicted labels.

- **Normalized Mutual Information (NMI)**  
  Measures the mutual dependence between the clustering result and the ground truth using entropy. This version uses **geometric mean normalization** for compatibility with most standard definitions.

---
