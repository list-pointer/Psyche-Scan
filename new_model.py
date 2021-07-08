# use pickle to save the trained model so that we dont have to train it again and again
import pickle
import random
# to choose who got the most votes
from statistics import mode

import nltk as nltk
# to inherit NLTK classifier
from nltk.classify import ClassifierI
# scikitlearn classfied as basic nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews
from sklearn.linear_model import LogisticRegression, SGDClassifier
# actual sklearn
# MultinominalNB should be very useful to us since we have multiple categories
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC


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
        conf = choice_votes / len(votes)
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

classfier_accuracy = []


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

basicNB_acuuracy = nltk.classify.accuracy(classifier, testing_set) * 100
# checking accuracy
print("Basic NB Classifier accuracy percent:", basicNB_acuuracy)
classfier_accuracy.append(basicNB_acuuracy)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
multinomialNB_accuraccy = nltk.classify.accuracy(MNB_classifier, testing_set) * 100
classfier_accuracy.append(multinomialNB_accuraccy)
print("MultinomialNB accuracy percent:", multinomialNB_accuraccy)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
bnb_accuracy = nltk.classify.accuracy(BNB_classifier, testing_set) * 100
classfier_accuracy.append(bnb_accuracy)
print("BernoulliNB accuracy percent:", bnb_accuracy)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
logistic_accuracy = nltk.classify.accuracy(LogisticRegression_classifier, testing_set) * 100
classfier_accuracy.append(logistic_accuracy)
print("LogisticRegression_classifier accuracy percent:", logistic_accuracy)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
sgdc_accuracy = nltk.classify.accuracy(SGDClassifier_classifier, testing_set) * 100
classfier_accuracy.append(sgdc_accuracy)
print("SGDClassifier_classifier accuracy percent:", sgdc_accuracy)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
svc_accuracy = nltk.classify.accuracy(SVC_classifier, testing_set) * 100
classfier_accuracy.append(svc_accuracy)
print("SVC_classifier accuracy percent:", svc_accuracy)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
linearSVC_accuracy = nltk.classify.accuracy(LinearSVC_classifier, testing_set) * 100
classfier_accuracy.append(linearSVC_accuracy)
print("LinearSVC_classifier accuracy percent:", linearSVC_accuracy)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
nuSVC_accuracy = nltk.classify.accuracy(NuSVC_classifier, testing_set) * 100
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

print("Voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",
      voted_classifier.confidence(testing_set[0][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",
      voted_classifier.confidence(testing_set[1][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",
      voted_classifier.confidence(testing_set[2][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",
      voted_classifier.confidence(testing_set[3][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",
      voted_classifier.confidence(testing_set[4][0]) * 100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",
      voted_classifier.confidence(testing_set[5][0]) * 100)

classfiers = ["Basic Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes Classier",
              "Logistic Regression", "SGDC", "SVC", "Linear SVC", "NuSVC"]

print("Classier Accuracy: ", classfier_accuracy)
print("Classifiers Implemented:", classfiers)

print("Cumulative Accuracy")
for i in range(len(classfiers)):
    print(classfiers[i], " : ", classfier_accuracy[i])

import matplotlib.pyplot as plt

plt.bar(range(8), classfier_accuracy)
plt.show()


