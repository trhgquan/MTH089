# TREC classification

## How-to
For `ml-models` scripts:

1. Download data (Train and Test):
```
bash download.sh
```

2. Run `dataprep.py`

## Results

| Model                                      | Accuracy     | Precision (macro) | Recall (macro) | F1 score (macro) |
| ------------------------------------------ | ------------ | ----------------- | -------------- | ---------------- |
| Logistic Regression                        | 0.852000     | 0.830745          | 0.897112       | 0.856029         |
| Multinomial Naive Bayes                    | 0.832000     | 0.703944          | 0.699418       | 0.696869         |
| Support Vector Classifier                  | 0.886000     | 0.862294          | 0.912049       | 0.882370         |
| Multilayer Perceptron (MLP)                | 0.828000     | 0.863599          | 0.788308       | 0.813223         |
| Bidirectional Gradient Gated Units (BiGRU) | 0.836000     | 0.694588          | 0.708307       | 0.700427         |
| Finetuned DistilBERT                       | **0.974000** | **0.976173**      | **0.977431**   | **0.976423**     |
| Finetuned XLM-RoBERTa                      | 0.966000     | 0.971092          | 0.969782       | 0.970092         |

### Logistic Regression
I used `sklearn.linear_model.LogisticRegression` with `penalty="l2"`. The pipeline include
- `CountVectorizer` with `n_grams = (1, 2)`
- `TfidfTransformer` with `use_idf = True`
- `LogisticRegression(penalty="l2")`

### Multinomial Naive Bayes
I used `sklearn.naive_bayes.MultinomialNB`. The pipeline include
- `CountVectorizer` with `n_grams = (1, 2)`
- `TfidfTransformer` with `use_idf = True`
- `MultinomialNB`

### Support Vector Machine
I used `sklearn.svm.LinearSVC`. The pipeline include
- `CountVectorizer` with `n_grams = (1, 2)`
- `TfidfTransformer` with `use_idf = True`
- `LinearSVC`

### Multilayer Perceptron (MLP)
A deep neural network (DNN), but only have 2 fully connected hidden layer.

I trained with these hyperparameters:
- `embedding_size = 64` (first layer input dimension) and `128` (second layer input dimension)
- First layer with a `Dropout` rate `0.6`
- Second layer with activation `Sigmoid`
- `loss = CrossEntropyLoss`
- `Adam` optimizer with `learning_rate = 0.001`, `StepLR` scheduler with `step_size = 1` and `gamma = 0.1`
- `clip_grad_norm_` with `max_norm = 0.1`
- `epochs = 100`
- `batch_size = 64`
- Early stopping with `10 epochs patience`

### Bidirectional Gated Gradient Units
I built a simple GRU network:
- Start with an embedding layer with `embedding_dim = 256`
- Dropdown layer with `dropdown_rate = 0.5`
- GRU `hidden_size = 512` and `num_layers = 4`
- `loss = CrossEntropyLoss`
- `Adam` optimizer with `learning_rate = 0.001`, `StepLR` scheduler with `step_size = 1` and `gamma = 0.1`
- `clip_grad_norm_` with `max_norm = 0.1`
- `epochs = 1000`
- `batch_size = 64`
- Early stopping with `20 epochs patience`

### Finetuned DistilBERT
I used [DistilBERT-base-uncased weights from huggingface](https://huggingface.co/distilbert-base-uncased), finetuning hyperparameters include
- `epochs = 5`
- `batch_size = 16`
- `weight_decay = .01`
- `learning_rate = 2e-5`

Training dataset is [TREC on Huggingface](https://huggingface.co/datasets/trec) (which is the same but operating as a Torch dataset). I split the original TREC training set to a training and validation set with a ratio of 8:2.

### Finetuned XLM-RoBERTa
I used [XLM-RoBERTa weights from huggingface](https://huggingface.co/xlm-roberta-base), finetuning hyperparameters include
- `epochs = 10`
- `batch_size = 16`
- `weight_decay = .01`
- `learning_rate = 2e-5`
- `early_stopping_patience = 3 (steps)`

Training data splitted like [Finetuned DistilBERT](#finetuned-distilbert)