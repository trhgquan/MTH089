max_seq_len = 70
example = _create_examples(input_text, labels)

pretrain_tokenizer = AutoTokenizer.from_pretrained("demdecuong/vihealthbert-base-word")

feature = convert_examples_to_features(example, max_seq_len=max_seq_len,tokenizer=pretrain_tokenizer,pad_token_label_id=0)

all_input_ids = torch.tensor([[f for f in feature[0].input_ids]], dtype=torch.long)

all_attention_mask = torch.tensor([[f for f in feature[0].attention_mask]], dtype=torch.long)

all_slot_labels_ids = torch.tensor([[f for f in feature[0].slot_labels_ids]], dtype=torch.long)