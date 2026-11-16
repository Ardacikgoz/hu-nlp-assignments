from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset
from nltk.corpus import stopwords
import string
import re
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import math
import warnings
from sklearn.exceptions import ConvergenceWarning
matplotlib.use('Qt5Agg')

warnings.filterwarnings("ignore", category=ConvergenceWarning)

dataset = load_dataset("imdb")

train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()


################# I have got help from AI in preprocess text functions #################
def remove_extra_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_punctuation(text):
    return remove_extra_whitespace(''.join(' ' if char in string.punctuation else char for char in text))


def remove_numbers(text):
    return remove_extra_whitespace(''.join(' ' if char.isdigit() else char for char in text))


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [' ' if word in stop_words else word for word in words]
    return remove_extra_whitespace(' '.join(filtered_words))

########################################################################################
def preprocess_text(text):
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = text.lower()
    text = remove_stopwords(text)
    text = remove_extra_whitespace(text)

    return text


def createCounter(text_series):
    all_words = " ".join(text_series).split()
    return Counter(all_words)


def bias_scores(trainInput):
    trainInput['text'] = trainInput['text'].apply(preprocess_text)
    train = trainInput

    posClass = train[train['label'] == 1]['text']
    negClass = train_df[train['label'] == 0]['text']

    posCounter = createCounter(posClass)
    negCounter = createCounter(negClass)
    totalCounter = createCounter(train['text'])

    biasScoresList = []

    for word in totalCounter:
        negVal = negCounter.get(word, 0)
        posVal = posCounter.get(word, 0)
        totalVal = posVal + negVal

        bias = abs(posVal - negVal) / totalVal * math.log(totalVal)

        biasScoresList.append((word, posVal, negVal, totalVal, bias))

    biasScoresList.sort(key=lambda x: (-x[4], x[0]))

    return biasScoresList[:10000]


top_words_with_scores = bias_scores(train_df)

top_words = [word for word, scores in top_words_with_scores]

vectorizer = CountVectorizer(vocabulary=top_words)

X_train = vectorizer.transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

y_train = train_df['label']
y_test = test_df['label']


train_accuracies = []
test_accuracies = []

for i in range(1, 26):
    model = LogisticRegression(max_iter=i)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)




plt.figure(figsize=(10, 6))
plt.plot(range(1, 26), train_accuracies, label='Train Accuracy')
plt.plot(range(1, 26), test_accuracies, label='Test Accuracy')
plt.xlabel("max_iter")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


""" In plot we can see that train accuracy is increasing with iteration value but int test accuracy has saturation
after 11th iteration test accuracy is not increasing wise versa it decreasing sometimes. So we can say that
we have reached max performance with changing iteration number."""

""" The preferences can change with the situation because of that I would use naive bayes when I need to get result 
faster like get an idea about data because we are computing values once and after that make calculations with them.
But in logistic regression it is more complicated and takes more time to process but it gives better results mostly."""