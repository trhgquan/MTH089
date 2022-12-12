# ConLL - NER

## How-to
Download data (Train and Test):
```
bash download.sh
```

## Results

| Model                                       | Accuracy     | Precision (weighted) | Recall (weighted) | F1 score (weighted) |
| ------------------------------------------- | ------------ | -------------------- | ----------------- | ------------------- |
| Conditional Random Field (CRF)              | 0.956434     | 0.956282             | 0.956434          | 0.956170            |
| Recurrent Neural Network (RNN)              | 0.858490     | 0.909933             | 0.858490          | 0.877241            |
| Gated Recurrent Units (GRU)                 | 0.889458     | 0.880343             | 0.889458          | 0.881657            |
| Bidirectional Gated Recurrent Units (BiGRU) | 0.917024     | 0.912772             | 0.917024          | 0.913364            |
| Finetuned DistilBERT                        | 0.973586     | 0.974048             | 0.973586          | 0.973668            |
| Finetuned DistilRoBERTa                     | 0.970184     | 0.969406             | 0.970184          | 0.969575            |
| Finetuned ALBERT                            | **0.974414** | **0.974137**         | **0.974414**      | **0.974211**        |
| Finetuned XLNet                             | 0.974403     | 0.973616             | 0.974403          | 0.973741            |

### Conditional Random Field (CRF)
I use Conditional Random Field with several rules, mostly defining a noun and personal / object names. CRF's parameters:
- `algorithms="lbfgs"`
- `c1 = 0.1`
- `c2 = 0.1`
- `max_iterations = 100`
- `all_possible_transitions = True`

### Recurrent Neural Network (RNN)
I build a simple RNN network with these hyperparameters:
- `embedding_dim = 64`
- `hidden_size = 64`
- `batch_size = 64`
- `epochs = 100`
- `learning_rate = .0001` with `Adam`
- Loss function: `NLLLoss()`
- Early stopping with `10 epochs patience`

### Gated Recurrent Units (GRU)
I replicated the RNN network above, but replaced RNN with GRU and using extra parameters:
- `num_layers = 4` - using 4 layers of GRU.

### Bidirectional Gated Recurrent Units (BiGRU)
A modified version of above architecture:
- `bidirectional = True`. This leads to `in_features = 2 * hidden_size` in last dense layer.
- `learning_rate = .001`.

### Finetuning DistilBERT
For the pretraining weights, I used [DistilBERT-base-uncased from huggingface](https://huggingface.co/distilbert-base-uncased). For the dataset, I use the [ConLL2003 dataset from huggingface](https://huggingface.co/datasets/conll2003) (which is still the same version of ConLL2003 NER dataset, but built as Torch Dataset). Configurations as below:
- `learning_rate =  2e-5`
- `batch_size = 16`
- `weight_decay = 0.01`
- `early_stopping_patience = 3 (steps)`
- `best_model_metrics = eval_loss`

Training stopped at 3600 steps.

## Finetuning DistilRoBERTa
[Original RoBERTa paper](https://arxiv.org/abs/1907.11692)

For the pretraining weights, I used [distilroberta-base](https://huggingface.co/distilroberta-base). I used the same configurations for DistilBERT finetuning.

## Finetuning ALBERT
[Original ALBERT paper](https://arxiv.org/abs/1909.11942)

For the pretraining weights, I used [albert-base-v2](https://huggingface.co/albert-base-v2). I used the same configurations for DistilBERT finetuning.

## Finetuning XLNet
[Original XLNet paper](https://arxiv.org/abs/1906.08237)

For the pretraining weights, I used [xlnet-base-cased](https://huggingface.co/xlnet-base-cased). I used the same configurations for DistilBERT finetuning.