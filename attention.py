from libs import * 

'''
    Blocks:
      1. Input Tokenization
      2. Positional Encoding
      3. Attention -------------- HERE
      4. Layer Normalization
      5. Feed Forward Network
      6. MLP layer
'''

'''
    Explaining the attention mechanism in steps:
        1. We choose our query vector (a word/subword from the input sequence) 
        2. We dot the query with every vector in the input sequence (to get the similarity between the chosen query and all the other words)
        3. We then pass the result of the dot product through a softmax function to get its projection to numbers between 0 and 1
        4. Then we multiply the weight from the previous step with its corresponding value vector from the input sequence
        5. Finally we add all the resulting values (weight * value vector) to get the importance of the query to all other vectors in the input sequence
'''

class Scaled_DotProduct_Attention(nn.Module):

    '''
        Input:  
        
            1. Batched sequence of token vectors; Dim(Batch Size, Sequence Length, Number of Channels(essentially the number of unique possible tokens))

        Output:
            1. Attention weights; Dim (Batch Size, Sequence Length, Sequence Length)
            2. The value vector after applying attention weights; Dim (Batch_Size, Sequence Length, Number of Channels)
    '''

    def __init__(self, n_embd, context_length, head_size):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * C ** -0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    
        weights = self.softmax(weights)

        v = self.value(x)
        score = weights @ v


        return score


class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, context_length, num_heads, head_size):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_embd)
        self.multiheads = nn.ModuleList([Scaled_DotProduct_Attention(n_embd, context_length, head_size) for _ in range(num_heads)])

    def forward(self, x):

        out = torch.cat([h(x) for h in self.multiheads], dim=-1) 
        out = self.proj(out)
        
        return out