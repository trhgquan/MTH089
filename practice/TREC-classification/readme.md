# TREC classification

## How-to
1. Download data (Train and Test):
```
bash download.sh
```

2. Run `dataprep.py`

## Statistics

| Model                                                     | Accuracy | Precision (macro) | Recall (macro) | F1 score (macro) |
| --------------------------------------------------------- | -------- | ----------------- | -------------- | ---------------- |
| Logistic Regression (TF-IDF & (1, 2) CountVectorizer)     | 0.852000 | 0.830745          | 0.897112       | 0.856029         |
| Multinomial Naive Bayes (TF-IDF & (1, 2) CountVectorizer) | 0.832000 | 0.703944          | 0.699418       | 0.696869         |
| SVC (TF-IDF & (1, 2) CountVectorizer)                     | **0.886000** | **0.862294**          | **0.912049**       | **0.882370**         |
