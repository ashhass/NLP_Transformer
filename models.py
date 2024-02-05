from libs import *
from trainer import *

class BigramModel(nn.Module):

    def __init__(self, vocab_size):
        super(BigramModel, self).__init__()
        self.process = Trainer(context_length=10, batch_size=32, file='input.txt')
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None: loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def train(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for steps in range(500):
            x, y = self.process.load_batch('train')

            logits, loss = self(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            print(loss.item())

        print(self.process.decode(self.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


process = Trainer(8, 32, 'input.txt')
inputs, targets = process.load_batch('train')
vocab_size = process.getDataLength()

bigram = BigramModel(vocab_size=vocab_size)
logits, loss = bigram(inputs, targets)

# generated_text = process.decode(bigram.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist())

bigram.train(bigram) 