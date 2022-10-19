import codecs
import pickle as pkl

def generate_examples(filepath):
    examples = []

    with codecs.open(filepath, "rb+") as f:
        for id_, row in enumerate(f):
            # ByteError in a sample
            label, _, text = row.replace(b"\xf0", b" ").strip().decode().partition(" ")

            coarse_label, _, fine_label = label.partition(":")

            examples.append((id_, {
                "label-coarse" : coarse_label,
                "label-fine" : fine_label,
                "text" : text,
            }))
    
    return examples

train = generate_examples("train_5500.label")
test = generate_examples("TREC_10.label")

# Train set
print(len(train))
print(train[0])

# Test set
print(len(test))
print(test[0])

# Create a dictionary for naive encoding

labels = [x["label-coarse"] for _, x in train]
print(len(labels))

# List of unique sentence label
set_labels = list(set(labels))
print(f"Labels set: {set_labels}")

word2idx = {set_labels[i] : i + 1 for i in range(len(set_labels))}
print(f"Word2Idx: {word2idx}")

idx2word = {i + 1 : set_labels[i] for i in range(len(set_labels))}
print(f"Idx2Word: {idx2word}")

# Dump data to files.
with open("train_corpus.pkl", "wb+") as f:
    pkl.dump(train, f)

with open("test_corpus.pkl", "wb+") as f:
    pkl.dump(test, f)

# Dump encoder to files.
with open("word2idx.pkl", "wb+") as f:
    pkl.dump(word2idx, f)

with open("idx2word.pkl", "wb+") as f:
    pkl.dump(idx2word, f)
