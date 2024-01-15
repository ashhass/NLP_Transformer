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



# Parts:
  # 1. Attention Mechanism
  # 2. Encoder-Decoder Block
  # 3. Layer Normalization
  # 4. FeedForward Networks
  # 5. Softmax Layer
  # 6. Embedding Vector


def attention(q, k, v):
  '''
    Return attention weights and the value matrix multiplied by attention weights. The value matrix contains learnt contextual information from the input sequence, and the attention weights weigh the different positions/vectors 
    according to their importance in predicting the next word.
  '''
  scores = (q * k.transpose(-2, -1)) / math.sqrt(query.size(-1))
  attn_weights = torch.softmax(scores)
  return torch.matmul(attn_weights, v), attn_weights
