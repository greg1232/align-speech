from gruut import sentences as gruut_sentences

def get_label_words(label):
    words = label.split()

    without_punctuation = [word for word in words if not is_punctuation(word)]

    normalized_words = []

    for word in without_punctuation:
        normalized_sentences = gruut_sentences(word)

        for sentence in normalized_sentences:
            for normalized_word in sentence:
                normalized_words.append(normalized_word.text.lower())

    normalized_words_no_punctuation = [word for word in normalized_words if not is_punctuation(word)]

    return normalized_words_no_punctuation

def is_punctuation(word):
    return word == "." or word == "," or word == "!" or word == "[" or word == "]"

