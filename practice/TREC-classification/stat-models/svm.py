import pickle as pkl
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

random.seed(42)


def main():
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
    svc_pipeline = Pipeline([
        ("count_vectorizer", CountVectorizer(ngram_range=(1, 2))),
        ("tfidf", TfidfTransformer(use_idf=True)),
        ("svc", LinearSVC())
    ])

    # Pick random sentence for testing
    example_test = random.sample(list(range(len(test_data))), 10)

    # Training SVM with tfidf
    svc_pipeline.fit(train_data, train_labels)

    print("Finished training - TFIDF")

    # Prediction
    predicted = svc_pipeline.predict(test_data)

    print(classification_report(predicted, test_labels))
    print(confusion_matrix(predicted, test_labels))
    print(f"Accuracy = {accuracy_score(predicted, test_labels):.6f}, Precision = {precision_score(predicted, test_labels, average = 'macro'):.6f}, Recall = {recall_score(predicted, test_labels, average = 'macro'):.6f}, F1 = {f1_score(predicted, test_labels, average = 'macro'):.6f}")
    print("Some examples from testing set")
    for i in example_test:
        print(
            f"{test_data[i]}\nPredicted: {idx2word[predicted[i]]}\nLabel: {idx2word[test_labels[i]]}")


if __name__ == "__main__":
    main()
