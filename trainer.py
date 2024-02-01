from tokenizer import Tokenizer
from libs import torch

class Trainer():

    def __init__(self, context_length, batch_size):
        super(Trainer, self).__init__() 
        self.context_length = context_length
        self.batch_size = batch_size
        self.tokenizer = Tokenizer()


    def load_data(self, input):
        with open(input, 'r') as f:
            data = f.read()
        return data


    def split_data(self, data):
        input = self.tokenize_data(self.load_data(data))['input_ids']
        size = int(0.9 * (len(input))) 
        train_data, val_data = input[ : size], input[size : ] 
        return torch.tensor(train_data), torch.tensor(val_data)


    def tokenize_data(self, data):
        x, y = data[ : self.context_length], data[1 : self.context_length + 1]
        
        for size in range(self.context_length):
            context = data[: size + 1] 
            target = data[size] 

        return self.tokenizer(data) 
    
    def load_batch(self, split, file):
        
        train_data, val_data = self.split_data(file)
        data = train_data if split == 'train' else val_data

        ix = torch.randint(len(data) - self.context_length, (self.batch_size,))
        x = torch.stack([data[i : self.context_length + i] for i in ix])
        y = torch.stack([data[i+1 : self.context_length + i + 1] for i in ix])
        print(x, y)

    def trainer(self):
        return 
        

trainer = Trainer(context_length = 10, batch_size = 8)
trainer.load_batch('train', 'input.txt')  