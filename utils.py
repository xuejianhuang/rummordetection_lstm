import pandas as pd
import matplotlib.pyplot as plt
import  torch
import jieba
import bz2
import  os,re
import gensim
import pkuseg
import numpy as np
from gensim.models import KeyedVectors   #gensim用来加载预训练词向量
import content_features_rnn.config as config





#解压预训练好的词向量bz2文件
def bz2Decompress():
    if os.path.exists ("./embeddings/sgns.weibo.bigram") == False:
        with open ("./embeddings/sgns.weibo.bigram", 'wb') as new_file, open ("./embeddings/sgns.weibo.bigram.bz2", 'rb') as file:
            decompressor = bz2.BZ2Decompressor ()
            for data in iter (lambda: file.read (100 * 1024), b''):
                new_file.write (decompressor.decompress (data))


#加载ced_dataset数据，转化成labes,contents
def get_df():
    weibo = pd.read_csv ('./data/ced_dataset.txt', sep='\t', names=['label', 'content'], encoding='utf-8')
    weibo = weibo.dropna ()  # 删除缺失值
    return  weibo['label'].values.tolist(),weibo['content'].values.tolist()

#jieba分词
def jieba_cut(contents):
    contents_S=[]
    for line in contents:
        current_segment=jieba.lcut(line)#列表，元素为分割出来的词
        contents_S.append (current_segment)
    return contents_S


#pkuseg分词，可以选择微博领域
def pkuseg_cut(contents,model_name="web"):
    seg = pkuseg.pkuseg (model_name='web')  # 程序会自动下载所对应的细领域模型
    contents_S = []
    for line in contents:
        line = re.sub ("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", line)
        current_segment = jieba.lcut (line)  # 列表，元素为分割出来的词
        contents_S.append(current_segment)
    return contents_S

#获取停用词集合
def get_stopwords():
    stopwords = pd.read_csv ("./stopwords/stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'],
                             encoding='utf-8')
    return  set(stopwords['stopword'].values.tolist())


#去除停用词
def drop_stopwords(contents,stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))    #str()转换为字符串##记录所有line_clean中的词
        contents_clean.append(line_clean)
    return contents_clean,all_words


def get_word2vec():
    word2vec=KeyedVectors.load_word2vec_format('./embeddings/sgns.weibo.bigram',binary=False,unicode_errors="ignore")
    return word2vec

## 返回预处理后的（labels，index)=>([1},[234 1234])
def key_to_index(contents,word2vec,num_words):
    '''
    :param contents:
    :param word2vec:预训练好的词向量模型，词向量根据使用频率降序排列
    :param num_words: 最大词汇数，选择使用前new_words个使用评率最高的词
    :return:
    '''
    train_tokens=[]
    contents_S = pkuseg_cut(contents)
    stopword = get_stopwords ()
    contents_clean, all_words = drop_stopwords (contents_S, stopword)
    for line_clean in contents_clean:
        for i, key in enumerate(line_clean):
            try:
                index=word2vec.key_to_index[key]
                if index<num_words:
                    line_clean[i]=word2vec.key_to_index[key]
                else:
                    line_clean[i] =0  #超出前num_words个词用0代替
            except KeyError:  #如果词不在字典中，则输出0
                line_clean[i]=0
        train_tokens.append(line_clean)
    return train_tokens

# 返回预处理后的（labels，contents)=>([1},[求 转发])
def labels_contents():
    labels, contents = get_df()
    contents_S = jieba_cut (contents)
    stopword = get_stopwords ()
    contents_clean, all_words = drop_stopwords (contents_S, stopword)
    contents_clean=[" ".join (x) for x in contents_clean]
    return labels, contents_clean

def get_maxlength(train_tokens):
    num_tokens = [len (tokens) for tokens in train_tokens]
    num_tokens = np.array (num_tokens)
    # 取tokens平均值并加上两个tokens的标准差，
    # 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本
    max_tokens = np.mean (num_tokens) + 2 * np.std (num_tokens)
    max_tokens = int (max_tokens)
    return max_tokens

def padding_truncating(train_tokens,maxlen):
    for i,token in enumerate(train_tokens):
        if len(token)>maxlen:
            train_tokens[i]=token[len(token)-maxlen:]
        elif len(token)<maxlen:
            train_tokens[i]=[0]*(maxlen-len(token))+token
    return train_tokens


def get_embedding(word2vec,num_words=50000,embedding_dim=300):
    '''
    :param num_words: 只选择使用前50k个使用频率最高的词
    :param embedding_dim: 词向量维度
    :param word2vec: 预训练好的词向量模型
    :return:
    '''
    embedding_matrix = np.zeros ((num_words, embedding_dim))
    # embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
    # 维度为 50000 * 300
    for i in range (num_words):
        # embedding_matrix[i,:] = cn_model[cn_model.index2word[i]]#前50000个index对应的词的词向量
        embedding_matrix[i, :] = word2vec[i]  # 前50000个index对应的词的词向量
    embedding_matrix = embedding_matrix.astype ('float32')
    return torch.from_numpy(embedding_matrix)

word2vec=get_word2vec()

embedding=get_embedding(word2vec,num_words=config.num_words,embedding_dim=config.embedding_dim)

def plot_learning_curve(train_loss,test_loss, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''

    plt.figure(figsize=(20, 8))
    plt.plot(train_loss, c='tab:red', label='train')
    plt.plot(test_loss, c='tab:cyan', label='test')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training epoch')
    plt.ylabel('CrossEntropyLoss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_learning_curve([1,2,3,4],[2,3,1,2])
    # labels,contents=labels_contents()
    # print(labels[:5])
    # print(contents[:5])
