from libs import * 
from attention import MultiHeadAttention

class MLP(nn.Module):

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
          )

  def forward(self, x):
    return self.net(x)  


class Block(nn.Module):

  def __init__(self, n_embd, context_length, num_heads):
    super().__init__()
    head_size = n_embd // num_heads
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    self.attention = MultiHeadAttention(n_embd, context_length, num_heads, head_size)
    self.MLP = MLP(n_embd)


  def forward(self, x):
    attn = x + self.attention(self.ln1(x))
    out = attn + self.MLP(self.ln2(attn))

    return out