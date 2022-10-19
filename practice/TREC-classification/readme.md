# TREC classification

## How-to
1. Download data (Train and Test):
```
bash download.sh
```

2. Run `dataprep.py`

## Stats

| Model                                     | Accuracy | Precision (macro) | Recall (macro) | F1 score (macro) |
| ----------------------------------------- | -------- | ----------------- | -------------- | ---------------- |
| Logistic Regression (CountVectorizer)     | 0.852000 | 0.832893          | 0.893321       | 0.854778         |
| Logistic Regression (TF-IDF)              | 0.852000 | 0.834321          | 0.888338       | 0.854695         |
| Multinomial Naive Bayes (CountVectorizer) | 0.760000 | 0.707486          | 0.804387       | 0.721629         |
| Multinomial Naive Bayes (TF-IDF)          | 0.760000 | 0.655242          | 0.645010       | 0.641378         |
| SVC (CountVectorizer)                     | 0.876000 | 0.853393          | 0.905282       | 0.873627         |
| SVC (TF-IDF)                              | 0.878000 | 0.859493          | 0.901867       | 0.875229         |
