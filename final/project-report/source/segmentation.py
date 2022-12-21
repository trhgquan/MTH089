target_VnCoreNLP_path = os.path.join(os.path.dirname(__file__), 'vncorenlp/VnCoreNLP-1.1.1.jar')
rdrsegmenter = VnCoreNLP(target_VnCoreNLP_path,
                         annotators="wseg",
                         max_heap_size='-Xmx500m')
sentences = rdrsegmenter.tokenize(text)
input_text = ""
for sentence in sentences:
    input_text = " ".join(sentence)