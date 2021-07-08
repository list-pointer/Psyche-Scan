# import pandas as pd
#
# df = pd.read_csv('dataset.csv')
#
# print(df.head(2))
#
# category = df['category'].unique()
#
# # category=set(category)
#
# print(type(category))
#
# print(category)

# import numpy as np
# import re
# import nltk
# from sklearn.datasets import load_files
# nltk.download('stopwords')
# import pickle
# from nltk.corpus import stopwords
#
# dataset = load_files(r"dataset.csv")
# X, y = dataset.data, dataset.target
#
# print(X)
# print(y)


# Data visual

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import warnings
# import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import chi2

sns.set_style("whitegrid")

# df = pd.read_csv("news.csv")
df = pd.read_csv("dataset.csv")

df.head(100)
# df.groupby('Category').count().Content[1]

########DATA VISUALIZATION PART##########


# label = ['business', 'entertainment', 'politics', 'sport', 'tech']
label = ['Visual', 'Kinesthetic', 'Auditory', 'Auditory Digital']
sum = 0
num_of_contents = []
for i in range(len(label)):
    num_of_contents.append(df.groupby('category').count().sentence[i])
    sum = sum + df.groupby('category').count().sentence[i]

num_of_contents

print(num_of_contents)


def plot_bar():
    plt.figure(figsize=(10, 6))
    index = np.arange(len(label))
    plt.bar(index, num_of_contents, color=['red', 'green', 'blue', 'orange'])
    plt.xticks(index, label, rotation=30)
    plt.xlabel('Categories')
    plt.ylabel('Number of Categories')
    plt.title('Number of contents in each Category')
    plt.show()


plot_bar()
#
perc_of_contents = []
for i in range(len(label)):
    perc_of_contents.append((df.groupby('category').count().sentence[i] / sum) * 100)


def plot_bar_perc():
    plt.figure(figsize=(10, 6))
    index = np.arange(len(label))
    plt.bar(index, perc_of_contents, color=['red', 'green', 'blue', 'orange', 'pink'])
    plt.xticks(index, label, rotation=30)
    plt.xlabel('Categories')
    plt.ylabel('Percentage of Categories')
    plt.title('Percentage of contents in each Category')
    plt.show()


plot_bar_perc()

df['sentence_length'] = df['sentence'].str.len()
df['sentence_length'][0]
#
plt.figure(figsize=(12.8, 6))
sns.distplot(df['sentence_length']).set_title('Sentence length distribution');
df['sentence_length'].describe()
plt.show()

quantile_95 = df['sentence_length'].quantile(0.95)
df_95 = df[df['sentence_length'] < quantile_95]

plt.figure(figsize=(12.8, 6))
sns.distplot(df_95['sentence_length']).set_title('Sentence length distribution');
plt.show()

df_more50 = df[df['sentence_length'] > 50]
len(df_more50)

df_more50['sentence'].iloc[0]

plt.figure(figsize=(12.8, 6))
sns.boxplot(data=df, x='category', y='sentence_length', width=.5);
plt.show()

# plt.figure(figsize=(12.8, 6))
# sns.boxplot(data=df_95, x='category', y='sentence_length');
# plt.show()

with open('sentence_dataset.pickle', 'wb') as output:
    pickle.dump(df, output)
# ########END DATA VISUALIZATION PART##########
#
#
# ########DATA CLEANING PART############
#
with open("sentence_dataset.pickle", 'rb') as data:
    df = pickle.load(data)

df.tail()

df['id'] = 1
df2 = pd.DataFrame(df.groupby('category').count()['id']).reset_index()
df.loc[1]['sentence']
df['temp1'] = df['sentence'].str.replace("\r", " ")
df['temp1'] = df['temp1'].str.replace("\n", " ")
df['temp1'] = df['temp1'].str.replace("    ", " ")

df['temp1'][1]
df['temp1'] = df['temp1'].str.replace('"', '')

df['temp2'] = df['temp1'].str.lower()

punctuation_signs = list("?:!.,;")
df['temp3'] = df['temp2']

for punct_sign in punctuation_signs:
    df['temp3'] = df['temp3'].str.replace(punct_sign, '')

df['temp4'] = df['temp3'].str.replace("'s", "")

wordnet_lemmatizer = WordNetLemmatizer()

nrows = len(df)
lemmatized_text_list = []

for row in range(0, nrows):

    lemmatized_list = []

    text = df.loc[row]['temp4']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

    lemmatized_text = " ".join(lemmatized_list)

    lemmatized_text_list.append(lemmatized_text)

df['temp5'] = lemmatized_text_list

stop_words = list(stopwords.words('english'))
df['cleaned_sentence'] = df['temp5']

for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    df['cleaned_sentence'] = df['cleaned_sentence'].str.replace(regex_stopword, '')

df.head()
print(df['sentence'][1])
print(df['cleaned_sentence'][1])
df['category'].head()


category_codes = {
    'Visual': 0,
    'Kinesthetic': 1,
    'Auditory': 2,
    'Auditory Digital': 3,
}
df['category_code'] = df['category']
df = df.replace({'category_code': category_codes})
# print(df.head())

# ########END DATA CLEANING PART############
#
# ########TRAIN TEST SPLIT##########
#
#
# # ecode = LabelEncoder()
# # df['E-Code'] = ecode.fit_transform(df['Category'])
df.head()
df.tail()
x_train, x_test, y_train, y_test = train_test_split(df['cleaned_sentence'], df['category_code'], test_size=0.15,
                                                    random_state=8)
x_train.count()
x_train.head()

# ######END TRAIN TEST SPLIT###########
#
# ########TFIDF###########
#
# Parameter election
ngram_range = (1, 2)
min_df = 10
max_df = 1.
max_features = 300

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

features_train = tfidf.fit_transform(x_train).toarray()
labels_train = y_train
print(features_train.shape)
features_train[0]

features_test = tfidf.transform(x_test).toarray()
labels_test = y_test
print(features_test.shape)

for Product, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")

bigrams

# X_train
with open('Pickles/x_train.pickle', 'wb') as output:
    pickle.dump(x_train, output)

# X_test
with open('Pickles/x_test.pickle', 'wb') as output:
    pickle.dump(x_test, output)

# y_train
with open('Pickles/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)

# y_test
with open('Pickles/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)

# df
with open('Pickles/df.pickle', 'wb') as output:
    pickle.dump(df, output)

# features_train
with open('Pickles/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('Pickles/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('Pickles/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('Pickles/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)

# TF-IDF object
with open('Pickles/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)

####END TFIDF###########


#########Training data##########


###########SVM################


########END SVM##############


# #####RANDOM FOREST ALGO###########
# '''mod=RandomForestClassifier(n_estimators=8)
# mod.fit(features_train,labels_train)
# predict = mod.predict(features_train)
# confusion_matrix(y_train,predict)
# accuracy_score(labels_train,predict)
# acc
#
#
#
#
#
cloud=WordCloud().generate(df['cleaned_sentence'][666])
plt.imshow(cloud)
plt.show()
# '''
# #############END RFA#############
