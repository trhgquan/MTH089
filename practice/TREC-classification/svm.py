import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report

# Load data
with open("train_corpus.pkl", "rb+") as f:
    train = pkl.load(f)

with open("test_corpus.pkl", "rb+") as f:
    test = pkl.load(f)

with open("word2idx.pkl", "rb+") as f:
    word2idx = pkl.load(f)

with open("idx2word.pkl", "rb+") as f:
    idx2word = pkl.load(f)

train_data = [x["text"] for _, x in train]
train_labels = [word2idx[x["label-coarse"]] for _, x in train]

test_data = [x["text"] for _, x in test]
test_labels = [word2idx[x["label-coarse"]] for _, x in test]

# Vectorizing training corpus
count_vectorizer = CountVectorizer(ngram_range = (1, 1))
X_train_counts = count_vectorizer.fit_transform(train_data)

tf_transformer = TfidfTransformer(use_idf = True).fit(X_train_counts)
X_train_tfidf = tf_transformer.transform(X_train_counts)

# Save transformer
with open("count.pkl", "wb+") as f:
    pkl.dump(X_train_counts, f)

with open("tfidf.pkl", "wb+") as f:
    pkl.dump(tf_transformer, f)

print("TFIDF vectorizer:")
print(X_train_tfidf.shape)
print(X_train_tfidf.toarray())

print("CountVectorizer")
print(X_train_counts.shape)
print(X_train_counts.toarray())

# Training SVM
svc = LinearSVC()
svc.fit(X_train_tfidf, train_labels)

print("Finished training")

# Prediction
X_test_counts = count_vectorizer.transform(test_data)
X_test_tfidf = tf_transformer.transform(X_test_counts)

predicted = svc.predict(X_test_tfidf)

print(classification_report(predicted, test_labels))
print(confusion_matrix(predicted, test_labels))