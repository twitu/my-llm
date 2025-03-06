# Solving for residual std scaling issue
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import pickle
import tiktoken
from tqdm import tqdm


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


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


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

    def total_batches(self):
        """Return the total number of batches in a full epoch"""
        return len(self.tokens) // (self.B * self.T)


def save_checkpoint(model, optimizer, epoch, loss, path="checkpoints"):
    """Save model checkpoint to file, keeping only the latest one"""
    if not os.path.exists(path):
        os.makedirs(path)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }

    # Always save to the same file name (overwriting previous checkpoints)
    checkpoint_path = os.path.join(path, "latest_checkpoint.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"Checkpoint saved to {checkpoint_path}")

    # For best model, use a separate file
    if path.endswith("/best"):
        print(f"New best model with loss: {loss:.4f}")


def load_checkpoint(model, optimizer=None, path="checkpoints"):
    """Load model checkpoint from file"""
    # Look for latest checkpoint
    checkpoint_path = os.path.join(path, "latest_checkpoint.pkl")

    if not os.path.exists(checkpoint_path):
        # Check if we're looking for best model but it doesn't exist
        if path.endswith("/best"):
            # Try to load from regular checkpoints instead
            return load_checkpoint(model, optimizer, "checkpoints")
        print(f"No checkpoint found at {checkpoint_path}")
        return None, 0, float("inf")

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {checkpoint_path}")
    return model, checkpoint["epoch"], checkpoint["loss"]


def train(
    model,
    train_loader,
    learning_rate,
    target_loss=0.0999,
    max_epochs=1000,
    resume=False,
):
    """Train the model until target loss is reached or max_epochs is hit"""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_epoch = 0
    best_loss = float("inf")

    # Ensure checkpoint directories exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/best", exist_ok=True)

    # Resume from checkpoint if requested
    if resume:
        model, start_epoch, best_loss = load_checkpoint(model, optimizer)
        if start_epoch is None:
            start_epoch = 0
        start_epoch += 1  # Start from the next epoch

    print(f"Starting training from epoch {start_epoch}")
    print(f"Training until loss < {target_loss} or {max_epochs} epochs reached")

    # Calculate total batches for progress tracking
    total_batches = train_loader.total_batches()
    batches_per_epoch = total_batches  # Use ALL batches in your dataset

    epoch = start_epoch
    current_loss = float("inf")

    # Train until target loss is reached or max_epochs is hit
    with tqdm(total=max_epochs, desc="Epochs", initial=start_epoch) as epoch_pbar:
        while epoch < max_epochs and current_loss > target_loss:
            total_loss = 0
            batch_count = 0

            # Training loop for this epoch
            with tqdm(
                total=batches_per_epoch, desc=f"Epoch {epoch} batches"
            ) as batch_pbar:
                for i in range(batches_per_epoch):
                    x, y = train_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    logits, loss = model(x, y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    batch_count += 1

                    # Update batch progress bar
                    batch_pbar.set_postfix(loss=f"{loss.item():.4f}")
                    batch_pbar.update(1)

            # Calculate average loss for the epoch
            current_loss = total_loss / batch_count

            # Update epoch progress bar
            epoch_pbar.set_postfix(loss=f"{current_loss:.4f}", best=f"{best_loss:.4f}")
            epoch_pbar.update(1)

            print(f"Epoch {epoch} completed. Average Loss: {current_loss:.4f}")

            # Save checkpoint (overwrite previous)
            save_checkpoint(model, optimizer, epoch, current_loss)

            # Save best model (if better than previous best)
            if current_loss < best_loss:
                best_loss = current_loss
                save_checkpoint(
                    model, optimizer, epoch, current_loss, path="checkpoints/best"
                )

            epoch += 1

    if current_loss <= target_loss:
        print(f"Target loss reached! Final loss: {current_loss:.4f}")
    else:
        print(f"Maximum epochs reached. Best loss: {best_loss:.4f}")

    return model


def generate_text(model, prompt=None, max_length=100, temperature=1.0):
    """Generate text using the trained model"""
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")

    if prompt:
        tokens = enc.encode(prompt)
        x = torch.tensor(tokens).unsqueeze(0).to(device)  # Shape: [1, prompt_length]
    else:
        # Start with a single token (could be random or a special start token)
        x = torch.tensor([[50256]]).to(device)  # Usually <|endoftext|> in GPT-2

    generated_tokens = x.tolist()[0]

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generation progress bar
    with tqdm(total=max_length, desc="Generating") as pbar:
        with torch.no_grad():
            for _ in range(max_length):
                # Forward the model to get logits
                logits, _ = model(x)

                # Get the logits for the last token
                logits = logits[:, -1, :] / temperature

                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                # Add the token to the sequence
                x = torch.cat((x, next_token), dim=1)

                # Store the new token
                generated_tokens.append(next_token.item())

                # Update progress bar
                pbar.update(1)

                # Optional: Break if end token is generated
                if next_token.item() == 50256:  # <|endoftext|>
                    break

    # Decode the generated tokens
    generated_text = enc.decode(generated_tokens)
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPT Language Model Training and Generation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "generate"],
        help="Mode: train or generate",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=32, help="Sequence length")
    parser.add_argument(
        "--target_loss", type=float, default=0.0999, help="Target loss to reach"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=1000, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--generate_length", type=int, default=100, help="Length of generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Text prompt for generation"
    )

    args = parser.parse_args()

    # Determine the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    if args.mode == "train":
        # Initialize model
        model = GPT(GPTConfig())

        # Initialize data loader
        train_loader = DataLoaderLite(B=args.batch_size, T=args.seq_length)

        # Train the model
        train(
            model, train_loader, args.lr, args.target_loss, args.max_epochs, args.resume
        )

    elif args.mode == "generate":
        # Initialize model
        model = GPT(GPTConfig())

        # Load the best model
        model, _, _ = load_checkpoint(model, path="checkpoints/best")
        if model is None:
            model, _, _ = load_checkpoint(model)
            if model is None:
                print("No checkpoint found. Using untrained model.")

        # Generate text
        generated_text = generate_text(
            model, args.prompt, args.generate_length, args.temperature
        )
        print("\nGenerated Text:")
        print(generated_text)
