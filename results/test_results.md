| Dataset     | Model           | Params          | Accuracy | Precision | Recall | F1 Score |
|-------------|-----------------|-----------------|----------|-----------|--------|----------|
| rt-polarity | Dummy |  | 0.5000 | 0.2500 | 0.5000 | 0.3333 |
| rt-polarity | SVM | C=1.4063384522657325, kernel=rbf | 0.6300 | 0.6321 | 0.6300 | 0.6285 |
| rt-polarity | Random Forest | n_estimators=30 | 0.6138 | 0.6145 | 0.6138 | 0.6131 |
| sephora | Dummy |  | 0.6350 | 0.4032 | 0.6350 | 0.4932 |
| sephora | SVM | C=0.5207578030989397, kernel=linear | 0.6669 | 0.6083 | 0.6669 | 0.5807 |
| sephora | Random Forest | n_estimators=60 | 0.6517 | 0.5840 | 0.6517 | 0.5916 |