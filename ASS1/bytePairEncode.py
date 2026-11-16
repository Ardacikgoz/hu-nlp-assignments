"""
Assignment 1 - Code template for AIN442/BBM497

@author: İsmail Furkan Atasoy
"""
import codecs
def initialVocabulary():
    
    # You can use this function to create the initial vocabulary.
    
    return list("abcçdefgğhıijklmnoöprsştuüvyzwxq"+
                "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ"+
                "0123456789"+" "+
                "!'^#+$%&/{([)]=}*?\\_-<>|.:´,;`@€¨~\"é")


def spaceAndUnderBarTokenizer(text): #adding space and underscore
    textSplit = text.split()
    newList = [[" "] + list(element) + ["_"] for element in textSplit]

    return newList

def findMostSeenPair(tokenized): # finding count of bigrams in tokenized list and returns most seen bigram and maxFreq if there is
    bigramDict = {}              # equivalance between bigrams on counts it compare according to ASCII and choose which is come first
    for array in tokenized:
        idx = 0
        while idx < len(array) - 1:
            bigram = (array[idx], array[idx+1])
            if bigram in bigramDict:
                bigramDict[bigram] += 1
            else:
                bigramDict[bigram] = 1
            idx += 1

    try:
        maxFreq = max(bigramDict.values())
        candidates = [bg for bg, count in bigramDict.items() if count == maxFreq] # list of max frequency bigrams
        selectedBigram = sorted(candidates)[0]
    except ValueError:  #when bigramDict is empty that means there is no another merge
        maxFreq = 0
        selectedBigram = None

    return selectedBigram,maxFreq

def changeTheArray(array,merge): # making merge operations on array
    idx = 0
    newArray = []
    while idx < len(array):
        if idx < len(array) - 1 and array[idx] + array[idx + 1] == merge:
            idx += 2 # to pass the merged item
            newArray.append(merge)
        else:
            newArray.append(array[idx])
            idx += 1
    return newArray



def bpeCorpus(corpus, maxMergeCount=10):
    # TO DO
    # You can refer to Example 1, 2 and 3 for more details.
    vocab = initialVocabulary()
    tokenized = spaceAndUnderBarTokenizer(corpus)
    counter = 0
    mergeList = []

    while counter < maxMergeCount:
        mostSeenBigram, maxFreq = findMostSeenPair(tokenized)

        if maxFreq <= 0: # means there is no other merge
            break

        mergedToken = mostSeenBigram[0] + mostSeenBigram[1] # string bigram to add vocab
        mergeList.append((mostSeenBigram, maxFreq))
        vocab.append(mergedToken)

        newTokenized = []
        for array in tokenized:
            newTokenized.append(changeTheArray(array,mergedToken))
        tokenized = newTokenized
        counter += 1

    Merges = mergeList
    Vocabulary = vocab
    TokenizedCorpus = tokenized
    return (Merges,Vocabulary,TokenizedCorpus)# Should return (Merges, Vocabulary, TokenizedCorpus)


def bpeFN(fileName, maxMergeCount=10):

    # TO DO
    # You can refer to Example 4 and 5 for more details.
    try:
        corpus = codecs.open(fileName, "r", "utf-8").read()
    except Exception as e:
        print("Error reading file:", e)

    Merges, Vocabulary, TokenizedCorpus = bpeCorpus(corpus, maxMergeCount)
    return Merges, Vocabulary, TokenizedCorpus# Should return (Merges, Vocabulary, TokenizedCorpus)

def bpeTokenize(str, merges):

    # TO DO
    # You can refer to Example 6, 7 and 8 for more details.

    tokenized = spaceAndUnderBarTokenizer(str)
    for merge in merges:
        newTokenized = []
        mergedStr = merge[0][0] + merge[0][1]
        for token in tokenized:
            newArr = changeTheArray(token, mergedStr)
            newTokenized.append(newArr)
        tokenized = newTokenized


    return tokenized# Should return the tokenized string as a list

def bpeFNToFile(infn, maxMergeCount=10, outfn="output.txt"):
    
    # Please don't change this function. 
    # After completing all the functions above, call this function with the sample input "hw01_bilgisayar.txt".
    # The content of your output files must match the sample outputs exactly.
    # You can refer to "Example Output Files" section in the assignment document for more details.
    
    (Merges,Vocabulary,TokenizedCorpus)=bpeFN(infn, maxMergeCount)
    outfile = open(outfn,"w",encoding='utf-8')
    outfile.write("Merges:\n")
    outfile.write(str(Merges))
    outfile.write("\n\nVocabulary:\n")
    outfile.write(str(Vocabulary))
    outfile.write("\n\nTokenizedCorpus:\n")
    outfile.write(str(TokenizedCorpus))
    outfile.close()

