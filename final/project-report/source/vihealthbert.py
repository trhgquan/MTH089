class ViHnBERT(RobertaPreTrainedModel):
    def __init__(self, config, args, slot_label_lst):
        super(ViHnBERT, self).__init__(config)
        self.args = args
        self.num_slot_labels = len(slot_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained bert

        self.slot_classifier = SlotClassifier(
            config.hidden_size,
            self.num_slot_labels,
            args.dropout_rate,
        )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)