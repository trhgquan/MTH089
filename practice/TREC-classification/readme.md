# TREC classification

## Dataset
We use TREC-6:
- 6 coarse class labels, 50 fine class labels.
- Training/test: 5500/500

The `trec` dataset can be download via [Huggingface](https://huggingface.co/datasets/trec) or using bash file:
```
bash download.sh
```

You might want to run `dataprep.py` before using any stats model.


## Experiments
### Logistic Regression
I used `sklearn.linear_model.LogisticRegression` with `penalty="l2"`. The pipeline include
- `CountVectorizer` with `n_grams = (1, 2)`
- `TfidfTransformer` with `use_idf = True`
- `LogisticRegression(penalty="l2")`

Implementation can be found [here](code/ml-models/logistic_regression.py)

### Multinomial Naive Bayes
I used `sklearn.naive_bayes.MultinomialNB`. The pipeline include
- `CountVectorizer` with `n_grams = (1, 2)`
- `TfidfTransformer` with `use_idf = True`
- `MultinomialNB`

Implementation can be found [here](code/ml-models/naive_bayes.py)

### Support Vector Machine
I used `sklearn.svm.LinearSVC`. The pipeline include
- `CountVectorizer` with `n_grams = (1, 2)`
- `TfidfTransformer` with `use_idf = True`
- `LinearSVC`

Implementation can be found [here](code/ml-models/svm.py)

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

Implementation can be found [here](code/deep-learning/mlp-classification.ipynb) (PyTorch)

### CNN

#### CNN for text classification
Follow [the article of chriskhanhtran](https://chriskhanhtran.github.io/posts/cnn-sentence-classification/), I created a CNN for text classification (mostly use chriskhanhtran's architecture and feeding data to it only). There are 3 versions available:
- Random embedding: using randomized (trainable) embedding vectors.
- Static embedding: using [FacebookAI's fastText](https://fasttext.cc/) for embedding, weights were frozen.
- Non-static embedding: still using fastText, but weights are trainable.

Configurations were kept as the original.

Implementation can be found [here](code/deep-learning/CNN-classification.ipynb) (PyTorch)

#### Multi-channel CNN with LSTM
Based on the paper [Multi-channel LSTM-CNN model for Vietnamese sentiment analysis](https://www.researchgate.net/publication/321259272_Multi-channel_LSTM-CNN_model_for_Vietnamese_sentiment_analysis), I implemented the same architecture for text classification. Configurations include
- `filter_sizes = [3, 4, 5]`
- `num_filters = [150, 150, 150]`
- `hidden_units = 128`
- `embed_dim = 200`
- `hidden_dim = 100`
- `dropout = .2`
- `learning_rate = .25`
- `epochs = 50`

[The original implementation in Tensorflow Keras can be found here](https://github.com/ntienhuy/MultiChannel). My implementations including
- The original architecture from the paper (which I converted from TF to torch), can be found [here](code/deep-learning/multichannel-cnn-lstm/CNN-LSTM.ipynb).
- I then replaced the LSTM with bidirectional GRU, still using randomized embedding. The code can be found [here](code/deep-learning/multichannel-cnn-lstm/CNN-BiGRU.ipynb)
- A version using FacebookAI's fastText boosts the accuracy dramatically. The code can be found [here, for the static embedding](code/deep-learning/multichannel-cnn-lstm/CNN-BiGRU-fasttext-freezed.ipynb) and [here for the non-static embedding](code/deep-learning/multichannel-cnn-lstm/CNN-BiGRU-fasttext-trainable.ipynb).

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

Implementation can be found [here](code/deep-learning/bigru-classification.ipynb) (PyTorch)

### Finetuned DistilBERT
I used [DistilBERT-base-uncased weights from huggingface](https://huggingface.co/distilbert-base-uncased), finetuning hyperparameters include
- `epochs = 5`
- `batch_size = 16`
- `weight_decay = .01`
- `learning_rate = 2e-5`

Training dataset is [TREC on Huggingface](https://huggingface.co/datasets/trec) (which is the same but operating as a Torch dataset). I split the original TREC training set to a training and validation set with a ratio of 8:2.

Implementation can be found [here](code/deep-learning/BERT-based-classification.ipynb)

### Finetuned XLM-RoBERTa
I used [XLM-RoBERTa weights from huggingface](https://huggingface.co/xlm-roberta-base), finetuning hyperparameters include
- `epochs = 10`
- `batch_size = 16`
- `weight_decay = .01`
- `learning_rate = 2e-5`
- `early_stopping_patience = 3 (steps)`

Training data splitted like [Finetuned DistilBERT](#finetuned-distilbert)

Implementation can be found [here](code/deep-learning/XLM-RoBERTa-based-classification.ipynb)

## Results

| Model                                                                                 | Accuracy     | Precision (macro) | Recall (macro) | F1 score (macro) |
| ------------------------------------------------------------------------------------- | ------------ | ----------------- | -------------- | ---------------- |
| [Logistic Regression](#logistic-regression)                                           | 0.852000     | 0.830745          | 0.897112       | 0.856029         |
| [Multinomial Naive Bayes](#multinomial-naive-bayes)                                   | 0.832000     | 0.703944          | 0.699418       | 0.696869         |
| [Support Vector Classifier](#support-vector-machine)                                  | 0.886000     | 0.862294          | 0.912049       | 0.882370         |
| [Multilayer Perceptron (MLP)](#multilayer-perceptron-mlp)                             | 0.828000     | 0.863599          | 0.788308       | 0.813223         |
| [Bidirectional Gradient Gated Units (BiGRU)](#bidirectional-gated-gradient-units)     | 0.836000     | 0.694588          | 0.708307       | 0.700427         |
| [CNN with random embedding (trainable)](#cnn-for-text-classification)                 | 0.726000     | 0.808797          | 0.684322       | 0.717606         |
| [CNN with fastText (freezed)](#cnn-for-text-classification)                           | 0.924000     | 0.932542          | 0.898361       | 0.911952         |
| [CNN with fastText (trainable)](#cnn-for-text-classification)                         | 0.910000     | 0.922051          | 0.885470       | 0.899350         |
| [Multi-channel CNN with LSTM](#multi-channel-cnn-with-lstm)                           | 0.652000     | 0.717115          | 0.636954       | 0.630829         |
| [Multi-channel CNN with BIGRU and random embedding](#multi-channel-cnn-with-lstm)     | 0.676000     | 0.716941          | 0.700641       | 0.689064         |
| [Multi-channel CNN with BiGRU and fastText (freezed)](#multi-channel-cnn-with-lstm)   | 0.914000     | 0.919263          | 0.895741       | 0.903690         |
| [Multi-channel CNN with BiGRU and fastText (trainable)](#multi-channel-cnn-with-lstm) | 0.902000     | 0.891531          | 0.884072       | 0.885555         |
| [Finetuned DistilBERT](#finetuned-distilbert)                                         | **0.974000** | **0.976173**      | **0.977431**   | **0.976423**     |
| [Finetuned XLM-RoBERTa](#finetuned-xlm-roberta)                                       | 0.966000     | 0.971092          | 0.969782       | 0.970092         |

### Discussions
- Transformer-based models achieve best results (by the power of self-attention, a large corpus used during the pretraining, ..etc.). 
- Using pre-trained word embedding will boost metrics dramatically. Most CNN implementations were *shit* until plugging a pre-trained word embedding in, which results in approximately +10% accuracy at most. 
- Switching from unidirectional seq2seq to bidirectional seq2seq **does** create a minor effect on the result. 