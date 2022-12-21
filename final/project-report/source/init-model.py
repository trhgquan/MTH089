with torch.no_grad():
        inputs = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "slot_labels_ids": all_slot_labels_ids,
        }

parser = argparse.ArgumentParser()
parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
args = parser.parse_args()

config = RobertaConfig.from_pretrained('demdecuong/vihealthbert-base-word', finetuning_task='')

target_model_save_path = os.path.join(os.path.dirname(__file__), 'model-save')

model = ViHnBERT.from_pretrained(
    target_model_save_path,
    config=config,
    args=args,
    slot_label_lst=labels,
)
outputs = model(**inputs)