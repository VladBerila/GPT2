from dataclasses import dataclass
import math
import os
import time
import torch
import torch.distributed
import torch.nn as nn
from torch.nn import functional as F
import inspect
import tiktoken

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0 # scale initialization, see https://arxiv.org/abs/2002.05202
        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate key, query, value for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (materializes the large (T, T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim = -1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # use flash attention for better performance and memory usage.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        #output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0 # scale initialization

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50000 BPE merges + 256 byte tokends + 1 end-of-text token
    n_layer: int = 12 # number of transformer layers
    n_head: int = 12 # number of attention heads
    n_embed: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        #init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * module.NANOGPT_SCALE_INIT) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        # forward the token and position embeddings
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of transformer layers
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embed are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print("using fused AdamW: ", use_fused)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard files
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, "no data found"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        #state
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        # advance the position
        self.current_position += B * T * self.num_processes
        # reset the position if we hit the end
        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y
# ---------------------------------------------

device = "cpu"
from torch.distributed import init_process_group, destroy_process_group
# det up DDP (DistributedDataParallel) if available

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # NCCL backend is just on Linux and just for NVIDIA GPUs
    # backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    backend = 'gloo'
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    print("ddp rank: ", ddp_rank)
    print("ddp local rank: ", ddp_local_rank)
    print("ddp world size: ", ddp_world_size)
    if torch.cuda.is_available():
        device = f'cuda:{ddp_local_rank}'
    else:
        device = f'mps:{ddp_local_rank}'
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    #autodetect the device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

print("using device: ", device)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(1337)

total_batch_size = 524288 # 2^19, ~0.5M tokens
B = 8 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total batch size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print("total desired batch size", total_batch_size)
    print("gradient accumulation steps", grad_accum_steps)

# get a data batch
train_loader = DataLoaderLite(B= B, T= T, process_rank= ddp_rank, num_processes= ddp_world_size, split = "train")
val_loader = DataLoaderLite(B= B, T= T, process_rank= ddp_rank, num_processes= ddp_world_size, split = "val")

torch.set_float32_matmul_precision('high') # TF32

# create the model
model = GPT(GPTConfig(vocab_size=50304)) # try with a more friendly vocab size (it divides by 2 multiple times)
model.to(device)
model = torch.compile(model) # speed improvement on Linux
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # if we are in ddp, get the underlying model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(it):
    # 1) linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min lr
    if it > max_steps:
        return min_lr
    # 3) cosine decay to min lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # starts at 1 and decays to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimization
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type = device)
for step in range(max_steps):
    t0 = time.time()

    # one in a while evaluate our validation loss
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_acum = 0
            val_loss_steps= 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_acum += loss
        if ddp:
            torch.distributed.all_reduce(val_loss_acum, op=torch.distributed.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_acum.item():.4f}")
        
    # once in a while generate from the model (except step 0 which is noise)
    # if step > 0 and step % 100 == 0:
    #     model.eval()
    #     num_return_sequences = 4
    #     max_length = 32
        
    #     # prefix tokens
    #     import tiktoken
    #     enc = tiktoken.get_encoding('gpt2')
    #     tokens = enc.encode("Hello, I am a language model, and I will complete this sentence:")
    #     tokens = torch.tensor(tokens, dtype=torch.long)
    #     tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    #     xgen = tokens.to(device)
    #     # generate
    #     sample_rng = torch.Generator(device=device)
    #     sample_rng.manual_seed(42 + ddp_rank)
    #     while xgen.size(1) < max_length:
    #         # forward the model
    #         with torch.no_grad():
    #             with torch.autocast(device_type=device, dtype=torch.bfloat16):
    #               logits, loss = model(xgen)
    #             # take the logits at the final step
    #             logits = logits[:, -1, :]
    #             # get the probabilities
    #             probs = F.softmax(logits, dim=-1)
    #             # do top-k sampling of 50
    #             topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    #             # select a token from the top-k probabilities
    #             ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
    #             # gather the corresponding indices
    #             xcol = torch.gather(topk_indices, -1, ix)
    #             # append to the sequence
    #             xgen = torch.cat((xgen, xcol), dim=1)

    #     # print the generated text
    #     for i in range(num_return_sequences):
    #         tokens = xgen[i, :max_length].tolist()
    #         decoded = enc.decode(tokens)
    #         print(f"rank {ddp_rank}, sample {i}: {decoded}")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps # normalize the loss
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # wait for the GPU to finish work
    if(device == "cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d}, loss: {loss_accum.item():.6f}, norm: {norm.item():.4f}, lr: {lr:.4f}, dt: {dt * 1000:.2f}ms, tokens/s: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()
