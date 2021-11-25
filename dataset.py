import torch
import utils as utils
from torch.utils.data import  Dataset,DataLoader
from sklearn.model_selection import train_test_split
import config as config
class RummorDataset(Dataset):

    def __init__(self,model="train"):
        super(RummorDataset, self).__init__()

        self.labels, self.contents =utils.get_df()
        self.contents=utils.key_to_index(self.contents,utils.word2vec,config.num_words)

        self.maxlen=utils.get_maxlength(self.contents)

        self.contents=utils.padding_truncating(self.contents,self.maxlen)

        x_train,x_test,y_trian,y_test=train_test_split(self.contents,self.labels,test_size=0.2,shuffle=True,random_state=0)
        if model=="train":
            self.contents=x_train
            self.labels=y_trian
        elif model=="test":
            self.contents = x_test
            self.labels = y_test

    def __getitem__(self, item):
        return torch.tensor(self.contents[item]),torch.tensor(self.labels[item])

    def __len__(self):
        return len(self.contents)

def get_dataloader(model="train"):
    dataset=RummorDataset(model=model)
    return DataLoader(dataset,batch_size=config.batch_size,shuffle=True if model=="train" else False)

if __name__ == '__main__':
    dataset=RummorDataset(model="trian")
    x,y=next(iter(dataset))
    print(x)
    print(y)
