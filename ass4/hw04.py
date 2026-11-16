import random
import gensim.downloader
import numpy

model = gensim.downloader.load("word2vec-google-news-300")


def replace_with_similar(sentence, indices):
    resultDict = {}
    splitText = sentence.split()

    wordsToChange = []

    for index in indices:
        wordsToChange.append(splitText[index])

    for word in wordsToChange:
        resultDict[word] = model.most_similar(word, topn=5)

    for i in range(len(splitText)):
        word = splitText[i]
        if word in wordsToChange:
            random_idx = random.randint(0, 4)
            splitText[i] = resultDict[word][random_idx][0]

    newSentence = " ".join(splitText)

    return newSentence, resultDict


def sentence_vector(sentence):
    splitSentence = sentence.split()
    vector_dict = {}

    for word in splitSentence:
        if word in model:
            vector_dict[word] = model[word][:300]
        else:
            vector_dict[word] = numpy.zeros(300)

    summedVector = numpy.zeros(300)
    for word_vec in vector_dict.values():
        summedVector += word_vec

    sentence_vec = summedVector / len(splitSentence)

    return vector_dict, sentence_vec


def most_similar_sentences(file_path, query):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]

    queryDict, queryVec = sentence_vector(query)
    listOfVecs = []

    for line in lines:
        lineDict, lineVec = sentence_vector(line)
        listOfVecs.append((line, lineVec))

    results = []
    for line, vec in listOfVecs:
        cosineSim = numpy.dot(queryVec, vec) / (numpy.linalg.norm(queryVec) * numpy.linalg.norm(vec))
        results.append((line, cosineSim))

    return sorted(results, key=lambda x: x[1], reverse=True)
