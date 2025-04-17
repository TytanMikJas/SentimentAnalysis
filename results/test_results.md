| Dataset     | Model           | Params          | Accuracy | Precision | Recall | F1 Score |
|-------------|-----------------|-----------------|----------|-----------|--------|----------|
| rt-polarity | dummy |  | 0.5000 | 0.2500 | 0.5000 | 0.3333 |
| rt-polarity | SVM | C=1.43604497971603, kernel=rbf | 0.6300 | 0.6321 | 0.6300 | 0.6285 |
| rt-polarity | Random Forest | n_estimators=30 | 0.6138 | 0.6145 | 0.6138 | 0.6131 |
| sephora | dummy |  | 0.6220 | 0.3869 | 0.6220 | 0.4770 |
| sephora | SVM | C=1.5, kernel=linear | 0.6550 | 0.6131 | 0.6550 | 0.5787 |
| sephora | Random Forest | n_estimators=30 | 0.6260 | 0.5424 | 0.6260 | 0.5563 |