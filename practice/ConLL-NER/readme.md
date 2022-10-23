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


### RNN
I build a simple RNN network with these hyperparameters:
- `embedding_dim = 64`
- `hidden_size = 64`
- `batch_size = 64`
- `epochs = 100`
- `learning_rate = .0001` with `Adam`
- Loss function: `NLLLoss()`
- Early stopping with `10 epochs patience`

## Statistics
| Model                    | Accuracy     | Precision (weighted) | Recall (weighted) | F1 score (weighted) |
| ------------------------ | ------------ | -------------------- | ----------------- | ------------------- |
| Conditional Random Field | **0.956434** | **0.956282**         | **0.956434**      | **0.956170**        |
| RNN                      | 0.858490     | 0.909933             | 0.858490          | 0.877241            |