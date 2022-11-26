def retrieve_symptoms(words, labels):
    list_symptoms = []
    current_label = []

    assert len(words) == len(labels)

    for i in range(len(words)):
        if labels[i] == "B-SYMPTOM_AND_DISEASE":
            if len(current_label) > 0:
                list_symptoms.append(" ".join(current_label).replace("_", " "))
            
            current_label = [words[i]]

        if labels[i] == "I-SYMPTOM_AND_DISEASE":
            current_label.append(words[i])

    return list_symptoms

def main():
    raw_text = "Thưa bác_sĩ , dạo này em bị ho có đờm , chóng_mặt , buồn_nôn , hay bị tiêu_chảy .".split(" ")
    labels = ["O", "O", "O", "O", "O", "O", "O", "B-SYMPTOM_AND_DISEASE", "I-SYMPTOM_AND_DISEASE", "I-SYMPTOM_AND_DISEASE", "O", "B-SYMPTOM_AND_DISEASE", "O", "B-SYMPTOM_AND_DISEASE", "O", "O", "O", "B-SYMPTOM_AND_DISEASE", "O"]

    symptoms_list = retrieve_symptoms(raw_text, labels)
    
    print(symptoms_list)

if __name__ == "__main__":
    main()