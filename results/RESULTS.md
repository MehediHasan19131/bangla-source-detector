# Results

## In-domain test (Regional-TinyStories held-out)

| Model | accuracy | f1 | macro_f1 | roc_auc | recall_synthetic | recall_translated |
|---|---|---|---|---|---|---|
| logreg_tfidf_word | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| linsvm_tfidf_char | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| banglabert | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| logreg_features | 0.904 | 0.901 | 0.904 | 0.958 | 0.927 | 0.880 |
| majority_baseline | 0.500 | 0.000 | 0.333 | nan | 1.000 | 0.000 |

## Out-of-domain (Bangla-Instruct vs alpaca-bengali)

| Model | accuracy | f1 | macro_f1 | roc_auc | recall_synthetic | recall_translated |
|---|---|---|---|---|---|---|
| logreg_features | 0.655 | 0.645 | 0.654 | 0.702 | 0.682 | 0.627 |
| banglabert | 0.522 | 0.676 | 0.382 | 0.788 | 0.046 | 0.998 |
| logreg_tfidf_word | 0.505 | 0.668 | 0.344 | 0.663 | 0.010 | 0.999 |
| linsvm_tfidf_char | 0.504 | 0.668 | 0.344 | 0.731 | 0.010 | 0.999 |
| majority_baseline | 0.500 | 0.000 | 0.333 | nan | 1.000 | 0.000 |
