def readCorpus(file_name):
    corpus = []
    with open(file_name, "r", encoding='utf-8') as f:
        line = f.readline()
        while line:
            corpus.append(line.strip().split('\t')[1])
            line = f.readline()
        
    normalized_words_set = set(corpus)
    return normalized_words_set

def normal_word(text):
    if len(text) == 0:
        return None
    if text == "O":
        return None
    if text[0] == '"':
        text = text[1:]
    if text[-1] == '"':
        text = text[:-1]
    return text