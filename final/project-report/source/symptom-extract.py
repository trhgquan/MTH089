def retrieve_symptoms(words, labels):
    list_symptoms = []
    current_label = []

    assert len(words) == len(labels)

    for i in range(len(words)):
        if labels[i] == "O":
            if len(current_label) > 0:
                list_symptoms.append(" ".join(current_label).replace("_", " "))
                current_label = ""

        if labels[i] == "B-SYMPTOM_AND_DISEASE":
            if len(current_label) > 0:
                list_symptoms.append(" ".join(current_label).replace("_", " "))

            current_label = [words[i]]

        if labels[i] == "I-SYMPTOM_AND_DISEASE":
            current_label.append(words[i])

    if len(current_label) > 0:
        list_symptoms.append(" ".join(current_label).replace("_", " "))

    return list_symptoms