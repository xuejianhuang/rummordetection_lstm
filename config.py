
import torch




num_words =50000  #最大词汇数，选择使用前new_words个使用评率最高的词
maxlength =58     #文本最大长度,utils包中的get_maxlength() 方法获得通过 ,取tokens平均值并加上两个tokens的标准差，
                                     # 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本

embedding_dim = 300 #词向量维度
lstm_hidden_size=128  #LSTM 隐含层size
lstm_num_layers=2     #LSTM 层数
lstm_bidirectional=True #LSTM是否为双向
lr=1e-3    #学习率

dropout = 0.2  #随机置零的概率
batch_size = 64
early_stop_cnt=6   #验证集准确率不提升时最多等待epoch，早停
epoch=30           #迭代次数


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
