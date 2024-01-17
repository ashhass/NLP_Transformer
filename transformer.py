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
      3. Attention
      4. Layer Normalization
      5. Feed Forward Network
      6. MLP layer
'''
def attention(q, k, v):
  d = q.size(-1)
  score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
  attn_weight = F.softmax(score, dim=-1)
  return torch.matmul(attn_weight, v), attn_weight
