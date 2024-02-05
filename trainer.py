from tokenizer import Tokenizer
from libs import torch

class Trainer():

    def __init__(self, context_length, batch_size, file):
        super(Trainer, self).__init__() 
        self.context_length = context_length
        self.batch_size = batch_size
        self.tokenizer = Tokenizer()
        self.file = file

    def load_data(self, input):
        with open(input, 'r') as f:
            data = f.read()
        return data

    def getDataLength(self):
        charSet = sorted(set(self.load_data(self.file)))
        return len(charSet)

    def split_data(self, data):
        input = self.encode()

        size = int(0.9 * (len(input))) 
        train_data, val_data = input[ : size], input[size : ] 
        return torch.tensor(train_data), torch.tensor(val_data)


    def tokenize_data(self, data):
        x, y = data[ : self.context_length], data[1 : self.context_length + 1]
        
        for size in range(self.context_length):
            context = data[: size + 1] 
            target = data[size] 

        return self.tokenizer(data) 

    def encode(self, string=None):
        chars = sorted(set(self.load_data(self.file)))
        stoi = {ch:i for i, ch in enumerate(chars)}

        encode = lambda s: [stoi[c] for c in s] 

        return encode(string) if string else encode(self.load_data(self.file))

    def decode(self, input):
        chars = sorted(set(self.load_data(self.file)))
        itos = {i:ch for i, ch in enumerate(chars)}
        decode = lambda s: ''.join([itos[x] for x in s])
        
        return decode(input)

    def load_batch(self, split):
        
        train_data, val_data = self.split_data(self.file)
        data = train_data if split == 'train' else val_data

        ix = torch.randint(len(data) - self.context_length, (self.batch_size,))
        x = torch.stack([data[i : self.context_length + i] for i in ix])
        y = torch.stack([data[i+1 : self.context_length + i + 1] for i in ix])
        
        return x, y

# process = Trainer(8, 4, 'input.txt')
# print(process.load_batch('train')) 