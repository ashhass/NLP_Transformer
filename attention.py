import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline

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
def attention(q, k, v, mask=None):
    d = q.size(-1)
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    # for positions where mask is 0 which are the positions we want to mask, set the score value to something really small so it has minimal contribution to the weight value
    if mask is not None:
        score = score.masked_fill(mask==0, -1000)
    
    attn_weight = F.softmax(score, dim=-1)
    
    return torch.matmul(attn_weight, v), attn_weight
