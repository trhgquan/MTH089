# ConLL - NER

## Dataset
I used the ConLL-2003 dataset:
- train/valid/test: 14041/3250/3453

The `conll2003` dataset can be download via [Huggingface](https://huggingface.co/datasets/conll2003) or using bash file:
```
bash download.sh
```

## Results

| Model                                                                                                             | Accuracy     | Precision (weighted) | Recall (weighted) | F1 score (weighted) |
| ----------------------------------------------------------------------------------------------------------------- | ------------ | -------------------- | ----------------- | ------------------- |
| [Conditional Random Field (CRF)             ](#conditional-random-field-crf)                                      | 0.956434     | 0.956282             | 0.956434          | 0.956170            |
| [Recurrent Neural Network (RNN)             ](#recurrent-neural-network-rnn)                                      | 0.858490     | 0.909933             | 0.858490          | 0.877241            |
| [Gated Recurrent Units (GRU)                ](#gated-recurrent-units-gru)                                         | 0.889458     | 0.880343             | 0.889458          | 0.881657            |
| [Bidirectional Gated Recurrent Units (BiGRU)](#bidirectional-gated-recurrent-units-bigru)                         | 0.917024     | 0.912772             | 0.917024          | 0.913364            |
| [Bidirectional Gated Recurrent Units (BiGRU) with fastText (freezed)](#bidirectional-gated-recurrent-units-bigru) | 0.940627     | 0.944152             | 0.940627          | 0.941875            |
| [Finetuned DistilBERT                       ](#finetuning-distilbert)                                             | 0.973586     | 0.974048             | 0.973586          | 0.973668            |
| [Finetuned DistilRoBERTa                    ](#finetuning-distilroberta)                                          | 0.970184     | 0.969406             | 0.970184          | 0.969575            |
| [Finetuned ALBERT                           ](#finetuning-albert)                                                 | **0.974414** | **0.974137**         | **0.974414**      | **0.974211**        |
| [Finetuned XLNet                            ](#finetuning-xlnet)                                                  | 0.974403     | 0.973616             | 0.974403          | 0.973741            |

### Discussion
- Transformer-based models are the most powerful architecture in this problem (Machine Translation, which is Transformer-based models' original mainstream task). ALBERT won the best metrics, compared to a **distilled** version of BERT and RoBERTa with some explanations including the model size and the model's pretraining tasks. 
- Again, using fastText can boost a model's performance. We have seen seq2seq climbing from 0.917 to 0.941 with fastText, hence using a pre-trained word embedding can have a massive impact to the model due to the semantics of the embedding layer.
- Using GRU (or LSTM) does have a decent impact on the seq2seq, since the RNN block suffers from gradient descent.

## Experiments
### Conditional Random Field (CRF)
I use Conditional Random Field with several rules, mostly defining a noun and personal / object names. CRF's parameters:
- `algorithms="lbfgs"`
- `c1 = 0.1`
- `c2 = 0.1`
- `max_iterations = 100`
- `all_possible_transitions = True`

Implemented code can be found [here](code/ml-models/ner-crf.ipynb)

### Recurrent Neural Network (RNN)
I build a simple RNN network with these hyperparameters:
- `embedding_dim = 64`
- `hidden_size = 64`
- `batch_size = 64`
- `epochs = 100`
- `learning_rate = .0001` with `Adam`
- Loss function: `NLLLoss()`
- Early stopping with `10 epochs patience`

Implemented code can be found [here](code/seq2seq/ner-rnn.ipynb)

### Gated Recurrent Units (GRU)
I replicated the RNN network above, but replaced RNN with GRU and using extra parameters:
- `num_layers = 4` - using 4 layers of GRU.

Implemented code can be found [here](code/seq2seq/ner-gru.ipynb)

### Bidirectional Gated Recurrent Units (BiGRU)
A modified version of above architecture:
- `bidirectional = True`. This leads to `in_features = 2 * hidden_size` in last dense layer.
- `learning_rate = .001`.

There are 2 versions:
- Using random embedding vectors: code can be found [here](code/seq2seq/ner-bigru.ipynb)
- Using FacebookAI's fastText: code can be found [here](code/seq2seq/ner-bigru-fasttext.ipynb) - metrics were improved dramatically.

### Finetuning DistilBERT
[Original DistilBERT paper](https://arxiv.org/abs/1910.01108)

For the pretraining weights, I used [DistilBERT-base-uncased from huggingface](https://huggingface.co/distilbert-base-uncased). For the dataset, I use the [ConLL2003 dataset from huggingface](https://huggingface.co/datasets/conll2003) (which is still the same version of ConLL2003 NER dataset, but built as Torch Dataset). Configurations as below:
- `learning_rate =  2e-5`
- `batch_size = 16`
- `weight_decay = 0.01`
- `early_stopping_patience = 3 (steps)`
- `best_model_metrics = eval_loss`

Training stopped at 3600 steps. Code can be found [here](code/transformers/ner-distil-bert.ipynb)

## Finetuning DistilRoBERTa
[Original RoBERTa paper](https://arxiv.org/abs/1907.11692)

For the pretraining weights, I used [distilroberta-base](https://huggingface.co/distilroberta-base). I used the same configurations for DistilBERT finetuning. Code can be found [here](code/transformers/ner-distil-roberta.ipynb)

## Finetuning ALBERT
[Original ALBERT paper](https://arxiv.org/abs/1909.11942)

For the pretraining weights, I used [albert-base-v2](https://huggingface.co/albert-base-v2). I used the same configurations for DistilBERT finetuning. Code can be found [here](code/transformers/ner-albert.ipynb)

## Finetuning XLNet
[Original XLNet paper](https://arxiv.org/abs/1906.08237)

For the pretraining weights, I used [xlnet-base-cased](https://huggingface.co/xlnet-base-cased). I used the same configurations for DistilBERT finetuning. Code can be found [here](code/transformers/ner-xlnet.ipynb)