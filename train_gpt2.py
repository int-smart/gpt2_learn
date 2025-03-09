from dataclasses import dataclass
import math
import inspect
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken
import numpy as np

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
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.config = config
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)  # [batch_size, seq_length, n_embd]

        # Break into heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # qkt = q @ k.transpose(-2, -1)
        # qkt = qkt / math.sqrt(k.size(-1))

        # # Make it autoregressive
        # qkt = qkt.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # attn_weights = nn.functional.softmax(qkt, dim=-1)

        # attn_out = attn_weights @ v

        # Using flash attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        return self.c_proj(attn_out)

def load_token(shard):
    npt = np.load(shard)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split="train"):
        self.shard_id = 0
        self.enc = tiktoken.encoding_for_model("gpt-2")
        self.batch_size = B
        self.context = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        self.shards = sorted([
            s for s in os.listdir(os.path.join(os.path.dirname(__file__), "edu_fineweb10B")) 
            if s.startswith(f"edufineweb_{split}_") and s.endswith(".npy")
        ])
        self.current_position = self.process_rank * B * T
        self.split = split
        self.data = load_token(os.path.join(os.path.dirname(__file__), f"edu_fineweb10B/{self.shards[self.shard_id]}"))
    
    def reset(self):
        self.shard_id = 0
        self.data = load_token(os.path.join(os.path.dirname(__file__), f"edu_fineweb10B/{self.shards[self.shard_id]}"))
        self.current_position = self.process_rank * self.batch_size * self.context
        
    def next_batch(self):
        buf = self.data[
            self.current_position : self.current_position
            + self.batch_size * self.context
            + 1
        ]
        x = buf[: self.batch_size * self.context].view(self.batch_size, self.context)
        y = buf[1 : self.batch_size * self.context + 1].view(
            self.batch_size, self.context
        )
        self.current_position += self.num_processes * self.batch_size * self.context
        if (
            self.current_position
            + self.num_processes * self.batch_size * self.context
            + 1
            > self.data.size(-1)
        ):
            self.shard_id = (self.shard_id + 1) % len(self.shards)
            self.data = load_token(os.path.join(os.path.dirname(__file__), f"edu_fineweb10B/{self.shards[self.shard_id]}"))
            self.current_position = self.process_rank * self.batch_size * self.context
        return x, y

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for i in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** (-0.5)
            torch.nn.init.normal_(module.weight, 0.0, std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict if p.requires_grad}

        decay_params = [p for pn, p in param_dict if len(p.size()) >= 2]
        nodecay_params = [p for pn, p in param_dict if len(p.size()) < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optim = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optim

    def forward(self, input, target):
        B, T = input.shape
        pos = torch.arange(0, T, dtype=torch.long, device=input.device)
        token_emb = self.transformer.wte(input)
        pos_emb = self.transformer.wpe(pos)

        x = token_emb + pos_emb
        for i in range(self.config.n_layer):
            x = self.transformer.h[i](x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # import code; code.interact(local=locals())
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss


ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
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
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    return text_data


# Set device
total_batch_size = 524288
B, T = 1, 1024
warmup_steps = 715
max_steps = 10e9 / total_batch_size  # For 1 epoch


def get_lr(it, max_lr, ratio):
    if it < warmup_steps:
        lr = max_lr * (it + 1) / warmup_steps
    elif it > max_steps:
        lr = ratio * max_lr

    theta = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = (1 + math.cos(theta * math.pi)) * 0.5
    lr = ratio * max_lr + coeff * (max_lr - ratio * max_lr)

    return lr


assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "make sure total_batch_size is divisible by B * T * ddp_world_size"
total_gradient_steps = total_batch_size // (B * T * ddp_world_size)

torch.set_float32_matmul_precision("high")

# Load dataset
trainloader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
)
validloader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="valid"
)

# Create log file
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")


model = GPT(GPTConfig())
model.to(device)

use_compile = (
    True  # torch.compile interferes with HellaSwag eval and Generation. TODO fix
)
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

# optimize!
optim = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device_type=device_type
)

for step in range(max_steps):
    start_time = time.time()
    last_step = (max_steps - 1 == step)

    if step % 250 == 0 or last_step:
        model.eval()

        with torch.no_grad():
            valid_loss_accum = 0.0
            valid_steps = 20
            for i in range(valid_steps):
                x, y = validloader.next_batch()
                x = x.to(device)
                y = y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            loss = loss / valid_steps
            valid_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(valid_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation ran at step {step} wit validation loss {valid_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"Validation ran at step {step} wit validation loss {valid_loss_accum.item():.4f}\n")
            
            if (step % 5000 == 0 or last_step) and step > 0:
                # Save model checkpoint
                chkpt_dir = os.path.join("checkpoints",f"step_{step}")
                os.makedirs(chkpt_dir, exist_ok=True)

                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'step': step,
                    'config': raw_model.config,
                    'valid_loss': valid_loss_accum.item()
                }

                checkpoint_path = os.path.join(chkpt_dir, "model.pt")
                torch.save(checkpoint, checkpoint_path)
                print(f"Model checkpoint saved at {checkpoint_path}")

    # Sample outputs
    if step % 250 == 0 or last_step:
        model.eval()
        max_length = 32
        num_return_sequences = 4
        sample_context = "Hello, I'm a language model,"
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        enc = tiktoken.encoding_for_model("gpt-2")
        context_tokens = enc.encode(sample_context)
        context_tensor = torch.tensor(context_tokens, dtype=torch.long)
        context_tensor = context_tensor.unsqueeze(0).repeat(num_return_sequences, 1)
        while context_tensor.size(-1) < max_length:
            with torch.no_grad():
                x = context_tensor.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _ = model(x, None)
                logits = logits[:,-1,:] # [B, T, vocab_size]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1, largest=True, sorted=True)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                id = torch.gather(topk_indices, -1, ix)
                context_tensor = torch.concat((context_tensor, id), -1)
        output = context_tensor.tolist()
        for sample in range(num_return_sequences):
            print(f"rank {ddp_rank} sample {sample}: {enc.decode(output[sample])}")
    
    # Train loop
    model.train()
    optim.zero_grad()
    loss_accum = 0.0

    # Set lr for all param groups
    lr = get_lr(step, 6e-4, 0.1)
    for param_group in optim.param_groups:
        param_group["lr"] = lr

    for gradient_step in range(total_gradient_steps):
        x, y = trainloader.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / total_gradient_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = gradient_step == total_gradient_steps - 1
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optim.step()
    end_time = time.time()
    if master_process:
        print(
            f"step {step}, lr {lr:.4f}, loss {loss.item()}, time {end_time - start_time:.4f}s, tokens/sec {(B*T)/(end_time - start_time):.4f}s, norm {norm:.4f}"
        )

if ddp:
    destroy_process_group()
