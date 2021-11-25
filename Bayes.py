import sklearn
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import  CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
import content_features_rnn.utils as utils

def text_feature(words,vectorizer="CountVectorizer",max_features=4000):
    if vectorizer=="CountVectorizer":
        vec = CountVectorizer (analyzer='word', max_features=max_features, lowercase=False)
        # 为words中每个词建立字典对应
    elif vectorizer=="TfidfVectorizer":
        vec = TfidfVectorizer (analyzer='word', max_features=max_features, lowercase=False)
    vec.fit(words)
    return vec

def bayes(vectorizer="CountVectorizer",max_features=4000):
    labels,contents=utils.labels_contents()
    vec=text_feature(contents,vectorizer=vectorizer,max_features=max_features)
    x_train,x_test,y_train,y_test=train_test_split(contents,labels,shuffle=True,test_size=0.1,random_state=0)
    classifier=MultinomialNB()
    classifier.fit(vec.transform(x_train),y_train)
    score=classifier.score(vec.transform (x_test), y_test)
    return score
if __name__ == '__main__':
    score=bayes()  #准确率87%
    print("CountVectorizer:",score)

