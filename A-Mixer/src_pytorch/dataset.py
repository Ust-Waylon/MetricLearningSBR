import os
import pickle
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, BatchSampler, SequentialSampler
from utils import split_validation, data_masks, Data

class SessionData:
    
    def __init__(self, data_dir='../../data', name='gowalla', validation=False, batch_size=100):
        self.validation = validation
        self.batch_size = batch_size
        self.data_path = os.path.join(data_dir, name)
        self.name = name
        self.setup()

    def setup(self):
        # Load training data
        train_path = f'../data/{self.name}/train.txt'
        test_path = f'../data/{self.name}/test.txt'
        
        self.train_data = pickle.load(open(train_path, 'rb'))
        if self.validation:
            self.train_data, self.valid_data = split_validation(self.train_data, 0.1)
        else:
            self.valid_data = pickle.load(open(test_path, 'rb'))
        
        self.train_data = Data(self.train_data, shuffle=True)
        self.valid_data = Data(self.valid_data, shuffle=True)
        
        # Load test data
        self.test_data = pickle.load(open(test_path, 'rb'))
        self.test_data = Data(self.test_data, shuffle=True)
    
    def train_dataloader(self):
        sampler = BatchSampler(SequentialSampler(self.train_data), batch_size=self.batch_size, drop_last=False)
        
        return DataLoader(self.train_data, sampler=sampler, num_workers=4, pin_memory=True)
    
    def val_dataloader(self):
        sampler = BatchSampler(SubsetRandomSampler(range(len(self.valid_data))), batch_size=self.batch_size, drop_last=False)

        return DataLoader(self.valid_data, sampler=sampler, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        sampler = BatchSampler(SubsetRandomSampler(range(len(self.test_data))), batch_size=self.batch_size, drop_last=False)

        return DataLoader(self.test_data, sampler=sampler, num_workers=4, pin_memory=True)
        

if __name__ == "__main__":

    data = SessionData()
    val_loader = data.val_dataloader()
    train_loader = data.train_dataloader()
    
    for i in train_loader:
        print(i)
        break 