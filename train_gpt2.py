from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_STEPS = 100
WARMUP_STEPS = 20

def get_lr(it, max_lr, ratio):
  if it < WARMUP_STEPS:
    lr = max_lr * (it+1) / WARMUP_STEPS
  else if it > MAX_STEPS:
    lr = ratio * max_lr

  theta = (it-WARMUP_STEPS)/(MAX_STEPS-WARMUP_STEPS)
  coeff = (1 + math.cos(theta*math.pi)) * 0.5
  lr = ratio * max_lr + coeff * (max_lr - ratio * max_lr)

  return lr

  
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
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
        self.c_proj.NANOGPT_SCALE_INIT = 1
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
        self.c_proj.NANOGPT_SCALE_INIT = 1
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

        # qkt = q @ k.transpose(-2, -1)
        # qkt = qkt / math.sqrt(k.size(-1))

        # # Make it autoregressive
        # qkt = qkt.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # attn_weights = nn.functional.softmax(qkt, dim=-1)

        # attn_out = attn_weights @ v

        # Using flash attention
        attn_out = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        return self.c_proj(attn_out)

import tiktoken
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split="train"):
        with open("/home/abhishek/Desktop/Brinc/Software/gpt2_learn/input.txt", 'r', encoding='utf-8') as f:
            text_data = f.read()
        enc = tiktoken.encoding_for_model("gpt-2")
        tokens = enc.encode(text_data)
        self.data = torch.tensor(tokens)
        self.batch_size = B
        self.context = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.current_position = self.process_rank * B * T

    def next_batch(self):
        buf = self.data[self.current_position:self.current_position+self.batch_size*self.context+1]
        x = buf[:self.batch_size*self.context].view(self.batch_size, self.context)
        y = buf[1:self.batch_size*self.context+1].view(self.batch_size, self.context)
        self.current_position += self.num_processes * self.batch_size * self.context
        if self.current_position + self.num_processes * self.batch_size*self.context + 1 > self.data.size(-1):
            self.current_position = self.process_rank * B * T
        return x, y

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
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if (hasattr(module, "NANOGPT_SCALE_INIT")):
                std *= (2*self.config.n_layer)**(-0.5)
            torch.nn.init.normal_(module.weight, 0.0, std)                
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)


    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict if p.requires_grad}

        decay_params = [p for pn, p in param_dict if len(p.size()) >= 2]
        nodecay_params = [p for pn, p in param_dict if len(p.size()) < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optim = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=use_fused)
        return optim

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
        # import code; code.interact(local=locals())
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss
    

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Get data
def get_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    return text_data

# Set device
total_batch_size = 524288
B,T = 1, 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
total_gradient_steps = total_batch_size // (B*T*ddp_world_size)

torch.set_float32_matmul_precision("high")

# Load dataset
loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")

model = GPT(GPTConfig())
model.to(device)

use_compile = True # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# optimize!
optim = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

import time
for epoch in range(700):
    start_time = time.time()
    optim.zero_grad()
    # Set lr for all param groups
    lr = get_lr(it, 3e-4, 0.1)
    for param_group in optim.param_groups:
      param_group['lr'] = lr

    for gradient_step in range(total_gradient_steps):
      x,y = loader.next_batch()
      x = x.to(device)
      y = y.to(device)
      with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x,y)
      loss = loss/total_gradient_steps
      loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optim.step()
    end_time = time.time()
    print(f"epoch {epoch}, lr {lr:.4f}, loss {loss.item()}, time {end_time - start_time:.4f}s, tokens/sec {(B*T)/(end_time - start_time):.4f}s, norm {norm:.4f}")

if ddp:
    destroy_process_group()