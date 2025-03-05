from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 4
    n_embd: int = 768

# transformer.h.0.ln_1.weight torch.Size([768])
# transformer.h.0.ln_1.bias torch.Size([768])
# transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.0.attn.c_attn.bias torch.Size([2304])
# transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.0.attn.c_proj.bias torch.Size([768])
# transformer.h.0.ln_2.weight torch.Size([768])
# transformer.h.0.ln_2.bias torch.Size([768])
# transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.0.mlp.c_fc.bias torch.Size([3072])
# transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.0.mlp.c_proj.bias torch.Size([768])

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x+self.attn(self.ln_1(x))
        x = x+self.mlp(self.ln_2(x))
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd*4)
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1) # [batch_size, seq_length, n_embd]

        # Break into heads
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs)

        qkt = q @ k.transpose(-2, -1)
        qkt = qkt / math.sqrt(k.size(-1))

        # Make it autoregressive
        qkt = qkt.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn_weights = nn.functional.softmax(qkt, dim=-1)

        attn_out = attn_weights @ v
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        return self.c_proj(attn_out)



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([
                Block(config) for i in range(config.n_layer)
            ]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
    

    def forward(self, input, target):
        B,T = input.shape
        pos = torch.arange(0,T, dtype=torch.long, device=input.device)
        token_emb = self.transformer.wte(input)
        pos_emb = self.transformer.wpe(pos)

        x = token_emb+pos_emb
        for i in range(self.config.n_layer):
            x = self.transformer.h[i](x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if target:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss
    
# Get data
def get_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    return text_data

# Set device
device ="cpu"

# Load dataset
text_data = get_text_data('input.txt')
import tiktoken
enc = tiktoken.encoding_for_model("gpt-2")
tokens = enc.encode(text_data)


buf = torch.tensor(tokens[:4*6+1])
buf = buf.to(device)

x = buf[:-1].view([4,6])
y = buf[1:].view([4,6])

model = GPT(GPTConfig())
model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(50):
    optim.zero_grad()
    logits, loss = model(x,y)
    loss.backward()
    optim.step()
    print(f"epoch {epoch}, loss {loss.item()}")
