"""
训练模型
"""
import torch
from content_features_rnn.lstm_model import LSTM_Model
from content_features_rnn.dataset import RummorDataset
import content_features_rnn.config as config
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from content_features_rnn.dataset import  get_dataloader
from torch.nn import CrossEntropyLoss
from content_features_rnn.utils import  plot_learning_curve
device = config.device

def train(epoch,model,loss_fn,optimizer,train_dataloader):
    model.train()
    loss_list = []
    train_acc = 0
    train_total = 0
    loss_fn.to(device)
    bar = tqdm(train_dataloader, total=len(train_dataloader))  #配置进度条
    for idx, (input, target) in enumerate(bar):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss =loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        loss_list.append (loss.cpu().item())
        optimizer.step()
        # 准确率
        output_max = output.max (dim=-1)  # 返回最大值和对应的index
        pred = output_max[-1]  # 最大值的index
        train_acc += pred.eq (target).cpu ().float ().sum ().item ()
        train_total += target.shape[0]
    acc = train_acc / train_total
    print("train epoch:{}  loss:{:.6f} acc:{:.5f}".format(epoch, np.mean(loss_list),acc))
    return acc,np.mean(loss_list)


def test(model,loss_fn,test_dataloader):
    model.eval()
    loss_list = []
    test_acc=0
    test_total=0
    loss_fn.to (device)
    with torch.no_grad():
        for input, target in test_dataloader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            loss_list.append(loss.item())
            # 准确率
            output_max = output.max(dim=-1) #返回最大值和对应的index
            pred = output_max[-1]  #最大值的index
            test_acc+=pred.eq(target).cpu().float().sum().item()
            test_total+=target.shape[0]
        acc=test_acc/test_total
        print("test loss:{:.6f},acc:{}".format(np.mean(loss_list), acc))
    return acc,np.mean(loss_list)


if __name__ == '__main__':
    model = LSTM_Model().to(device)
    count_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters:,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    train_dataloader = get_dataloader(model='train')
    test_dataloader = get_dataloader(model='test')
    loss_fn=CrossEntropyLoss()
    best_acc=0
    early_stop_cnt=0
    train_loss_list=[]
    test_loss_list=[]
    for epoch in range(config.epoch):
        train_acc,train_loss=train(epoch,model,loss_fn,optimizer,train_dataloader)
        test_acc,test_loss=test(model,loss_fn,test_dataloader)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        if test_acc>best_acc:
            best_acc=test_acc
            torch.save(model.state_dict(), 'model/model.pkl')
            print("save model,acc:{}".format(best_acc))
            early_stop_cnt=0
        else:
            early_stop_cnt+=1
        if early_stop_cnt>config.early_stop_cnt:
            break
    plot_learning_curve(train_loss_list,test_loss_list)

