'''
    Blocks:
      1. Input Encoder
      2. Positional Encoding
      3. Attention ----------- HERE
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

    def __init__(self):
        super(Scaled_DotProduct_Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None):
        d = q.size(-1)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    
        # for positions where mask is 0 which are the positions we want to mask, set the score value to something really small so it has minimal contribution to the weight value
        if mask is not None:
            score = score.masked_fill(mask==0, -1000)
        
        attn_weight = self.softmax(score)
        
        return torch.matmul(attn_weight, v), attn_weight
