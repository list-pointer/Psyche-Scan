# import nltk
# nltk.download()


# tokenizing
# from nltk.tokenize import sent_tokenize,word_tokenize
# EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
# print(sent_tokenize(EXAMPLE_TEXT))
# print(word_tokenize(EXAMPLE_TEXT))


# Stop words
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# example_sent = "This is a sample sentence, showing off the stop words filtration."
# stop_words = set(stopwords.words('english'))
# word_tokens = word_tokenize(example_sent)
# filtered_sentence = [w for w in word_tokens if not w in stop_words]
# filtered_sentence = []
# for w in word_tokens:
#     if w not in stop_words:
#         filtered_sentence.append(w)

# print(word_tokens)
# print(filtered_sentence)


# Stemming
# from nltk.stem import PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize
# ps = PorterStemmer()
# example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]
# for w in example_words:
#     print(ps.stem(w))
# new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
# words = word_tokenize(new_text)
# for w in words:
#     print(ps.stem(w))


# # Part of speech tagging
# import nltk
# # from nltk.corpus import state_union
# # from nltk.tokenize import PunktSentenceTokenizer
# # train_text = state_union.raw("2005-GWBush.txt")
# # # sample_text = state_union.raw("2006-GWBush.txt")
# # custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
# # tokenized = custom_sent_tokenizer.tokenize("Hello World! I am Shivam. Who are you? I work enthusiastically. ")
# # def process_content():
# #     try:
# #         for i in tokenized[:5]:
# #             words = nltk.word_tokenize(i)
# #             tagged = nltk.pos_tag(words)
# #             print(tagged)
# #     except Exception as e:
# #         print(str(e))
# # process_content()


# Chunking
# import nltk
# from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer
# train_text = state_union.raw("2005-GWBush.txt")
# # sample_text = state_union.raw("2006-GWBush.txt")
# custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
# tokenized = custom_sent_tokenizer.tokenize("Shivam like to work hard and he is not a good orator, pythonly coded but Abhishek is a smart worker as well as a nicely groomed orator.")
# def process_content():
#     try:
#         for i in tokenized:
#             words = nltk.word_tokenize(i)
#             tagged = nltk.pos_tag(words)
#             chunkGram = r"""Chunk: {<NNP>+<NN>?}"""
#             chunkParser = nltk.RegexpParser(chunkGram)
#             chunked = chunkParser.parse(tagged)
#             chunked.draw()
#     except Exception as e:
#         print(str(e))
# process_content()


# Chinking
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
train_text = state_union.raw("2005-GWBush.txt")
# sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize("Shivam like to work hard and he is not a good orator, pythonly coded but Abhishek is a smart worker as well as a nicely groomed orator.")
def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            # chunkGram = r"""Chunk: {<.*>+}"""
            chunkGram = r"""Chunk: {<.*>+}
                            }<CC>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
    except Exception as e:
        print(str(e))
process_content()


named entity recognition
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
train_text = state_union.raw("2005-GWBush.txt")
# sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize("I am Shivam Chaubey and I study in SPIT since 2019")
def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=False)
            namedEnt.draw()
    except Exception as e:
        print(str(e))
process_content()


lemetizing words --> like stemming but the returned word is actually a proper word. it could be synonym
v for verb a for adjective in 'pos(part of speech)' attribute default is noun
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("heard",pos="v"))
print(lemmatizer.lemmatize("writing", pos="v"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))


CORPUS
import nltk
print(nltk.__file__)
mostly in %appdata% would be the file for windows
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg
# sample text
sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
for x in range(5):
    print(tok[x])
# print(tok[0:5])


WordNet
with wordnet you can see definations, synonyms, antonyms even context
from nltk.corpus import wordnet
# synonyms
syns = wordnet.synsets("program")
print(syns)
# for getting the name of synset
print(syns[1].lemmas()[0].name())
for s in syns:
    print(s.lemmas()[0].name())
# definition
print(syns[0].definition())
# examples
print(syns[0].examples())
# making a synonyms and antonyms set
synonyms = []
antonyms = []
for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(set(synonyms))
print(set(antonyms))
# checking similarities
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))


text classifier Positive or negative movie review
import nltk
import random
from nltk.corpus import movie_reviews
# use pickle to save the trained modelso that we dont have to train it again and again
import pickle


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
#
# print(all_words["fabulous"])

word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

# training the classifier
# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# checking accuracy
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

# informative words
classifier.show_most_informative_features(15)

saving the classifier
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


incorporating scikit and other classifiers as well
import nltk
import random
from nltk.corpus import movie_reviews
# use pickle to save the trained modelso that we dont have to train it again and again
import pickle

# scikitlearn classfied as basic nltk
from nltk.classify.scikitlearn import SklearnClassifier

# actual sklearn
# MultinominalNB should be very useful to us since we have multiple categories
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
#
# print(all_words["fabulous"])

word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

# training the classifier
# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# checking accuracy
print("Basic NB Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

# informative words
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set)*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set)*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


# voting system
import nltk
import random

import nltk as nltk
from nltk.corpus import movie_reviews
# use pickle to save the trained model so that we dont have to train it again and again
import pickle

# scikitlearn classfied as basic nltk
from nltk.classify.scikitlearn import SklearnClassifier

# actual sklearn
# MultinominalNB should be very useful to us since we have multiple categories
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# to inherit NLTK classifier
from nltk.classify import ClassifierI
# to choose who got the most votes
from statistics import mode

# vote classifier class

class VoteClassfier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf
        


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
#
# print(all_words["fabulous"])

word_features = list(all_words.keys())[:3000]

classfier_accuracy=[]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

# training the classifier
# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# informative words
classifier.show_most_informative_features(15)


basicNB_acuuracy = nltk.classify.accuracy(classifier, testing_set)*100
# checking accuracy
print("Basic NB Classifier accuracy percent:",basicNB_acuuracy)
classfier_accuracy.append(basicNB_acuuracy)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
multinomialNB_accuraccy = nltk.classify.accuracy(MNB_classifier, testing_set)*100
classfier_accuracy.append(multinomialNB_accuraccy)
print("MultinomialNB accuracy percent:",multinomialNB_accuraccy)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
bnb_accuracy = nltk.classify.accuracy(BNB_classifier, testing_set)*100
classfier_accuracy.append(bnb_accuracy)
print("BernoulliNB accuracy percent:",bnb_accuracy)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
logistic_accuracy  = nltk.classify.accuracy(LogisticRegression_classifier, testing_set)*100
classfier_accuracy.append(logistic_accuracy)
print("LogisticRegression_classifier accuracy percent:",logistic_accuracy)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
sgdc_accuracy = nltk.classify.accuracy(SGDClassifier_classifier, testing_set)*100
classfier_accuracy.append(sgdc_accuracy)
print("SGDClassifier_classifier accuracy percent:", sgdc_accuracy)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
svc_accuracy = nltk.classify.accuracy(SVC_classifier, testing_set)*100
classfier_accuracy.append(svc_accuracy)
print("SVC_classifier accuracy percent:", svc_accuracy)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
linearSVC_accuracy = nltk.classify.accuracy(LinearSVC_classifier, testing_set)*100
classfier_accuracy.append(linearSVC_accuracy)
print("LinearSVC_classifier accuracy percent:", linearSVC_accuracy)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
nuSVC_accuracy = nltk.classify.accuracy(NuSVC_classifier, testing_set)*100
classfier_accuracy.append(nuSVC_accuracy)
print("NuSVC_classifier accuracy percent:", nuSVC_accuracy)

voted_classifier = VoteClassfier(classifier,
                                 MNB_classifier,
                                 BNB_classifier,
                                 LogisticRegression_classifier,
                                 SGDClassifier_classifier,
                                 SVC_classifier,
                                 LinearSVC_classifier,
                                 NuSVC_classifier)

print("Voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)

classfiers = ["Basic Naive Bayes","Multinomial Naive Bayes","Bernoulli Naive Bayes Classier",
              "Logistic Regression","SGDC","SVC","Linear SVC","NuSVC"]

print("Classier Accuracy: ",classfier_accuracy)
print("Classifiers Implemented:",classfiers)


print("Cumulative Accuracy")
for i in range(len(classfiers)):
    print(classfiers[i]," : ",classfier_accuracy[i])


import matplotlib.pyplot as plt
plt.bar(range(8), classfier_accuracy)
plt.show()


