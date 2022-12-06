# ConLL - NER

## How-to
Download data (Train and Test):
```
bash download.sh
```

## Methods
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
We use the [ConLL2003 dataset from huggingface](https://huggingface.co/datasets/conll2003). Configurations as below:
- `learning_rate` : 2e-5
- `batch_size` : 16
- `weight_decay` : .01
- `early_stopping_patience` : 3 (steps)
- `best_model_metrics` : `eval_loss`

Training stopped at 3600 steps.

## Statistics
| Model                                       | Accuracy     | Precision (weighted) | Recall (weighted) | F1 score (weighted) |
| ------------------------------------------- | ------------ | -------------------- | ----------------- | ------------------- |
| Conditional Random Field (CRF)              | 0.956434     | **0.956282**         | **0.956434**      | **0.956170**        |
| Recurrent Neural Network (RNN)              | 0.858490     | 0.909933             | 0.858490          | 0.877241            |
| Gated Recurrent Units (GRU)                 | 0.889458     | 0.880343             | 0.889458          | 0.881657            |
| Bidirectional Gated Recurrent Units (BiGRU) | 0.917024     | 0.912772             | 0.917024          | 0.913364            |
| Finetuned DistilBERT                        | **0.975054** | 0.889020             | 0.892713          | 0.890863            |


