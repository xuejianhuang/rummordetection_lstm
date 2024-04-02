# 基于文本特征的微博谣言检测
  传统的谣言检测模型一般根据谣言的内容、用户属性、传播方式人工地构造特征，而人工构建特征存在考虑片面、浪费人力等现象。本次实践使用基于循环神经网络（LSTM/GRU）的谣言检测模型，将谣言本文向量化，通过循环神经网络的学习训练来挖掘表示文本深层的特征，避免了特征构建的问题，并能发现那些不容易被人发现的特征，从而产生更好的效果。
  * 开发语言：python、pytorch、sklearn
# 数据集说明
  https://github.com/thunlp/Chinese_Rumor_Dataset, 基于中文微博谣言数据集构建了本项目的data/ced_dataset.txt数据集,文件中每一行代表一条微博信息（label,content），0：谣言，1：非谣
# 数据预处理
  * 使用pkuseg分词（微博领域），去除停用词；
  * 预训练词向量使用北京师范大学中文信息处理研究所与中国人民大学 DBIIR 实验室的研究者开源的"chinese-word-vectors"，使用微博语料训练出的词向量，github链接为：https://github.com/Embedding/Chinese-Word-Vectors；
  * 索引长度标准化：长度统一为$np.mean(num_tokens) + 2 * np.std(num_tokens)$。
 # 模型搭建
   * Bayes模型：基于Tfidf建立文本特征表示，然后基于Bayes构建分类模型
   * LSTM模型：基于词向量构建文本特征表示，然后基于LSTM构建分类模型
# 文件说明
  * data:数据集目录
  * embeddings:预训练好的词向量保存目录，下载地址：https://pan.baidu.com/s/11PWBcvruXEDvKf2TiIXntg
  * model:模型保持目录
  * stopwords:相关停用词列表目录
  * Bayes.py：基于sklearn构建Bayes分类模型
  * config.py:相关参数配置文件
  * dataset.py:构建pytorch训练时的dataset,dataloader
  * lstm_model.py:基于pytorch构建LSTM模型
  * train.py:模型训练
  * utils.py:工具包文件
 # 主要第三方包
   * numpy=1.21.4
   * pandas=1.2.0
   * matplotlib=3.3.2
   * sklearn=0.24.1
   * pytorch=1.9.1
   * pkuseg
   * gensim=4.0.1
# 实验结果
  * Bayes:测试集准确率87%左右
  * LSTM:训练集96%，测试集87%，存在过拟合，需要做相关参数调整，留个大家自己尝试
# 可能出现的问题
  * python3.9安装pkuseg可能会出错，可以安装spacy_pkuseg，import spacy_pkuseg as pkuseg
   
