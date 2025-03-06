# Solving for residual std scaling issue
import os
import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import pickle
import tiktoken
from tqdm import tqdm
from model import GPT, GPTConfig


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


def load_checkpoint(model, optimizer=None, path="checkpoints", map_location=None):
    """Load model checkpoint from file"""
    # Look for latest checkpoint
    checkpoint_path = os.path.join(path, "latest_checkpoint.pkl")

    if not os.path.exists(checkpoint_path):
        # Check if we're looking for best model but it doesn't exist
        if path.endswith("/best"):
            # Try to load from regular checkpoints instead
            return load_checkpoint(model, optimizer, "checkpoints", map_location)
        print(f"No checkpoint found at {checkpoint_path}")
        return None, 0, float("inf")

    # Use torch.load instead of pickle.load, with map_location parameter
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {checkpoint_path}")
    return model, checkpoint["epoch"], checkpoint["loss"]


def train(
    model,
    train_loader,
    learning_rate=3e-4,
    target_loss=0.0999,
    max_epochs=1000,
    weight_decay=0.1,
    resume=False,
):
    """Train the model with improved training techniques"""
    model.to(device)

    # Set high precision for matmul operations
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Setup optimizer with weight decay
    if hasattr(model, "configure_optimizers"):
        optimizer = model.configure_optimizers(
            weight_decay=weight_decay, learning_rate=learning_rate, device_type=device
        )
    else:
        # Fallback to simple optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8
        )

    # Learning rate scheduler setup
    max_lr = learning_rate
    min_lr = max_lr * 0.1
    warmup_epochs = 5  # Adapt based on your dataset size

    def get_lr(epoch, batch, total_batches):
        # Convert to iteration space
        it = epoch * total_batches + batch
        warmup_iters = warmup_epochs * total_batches
        max_iters = max_epochs * total_batches

        if it < warmup_iters:
            return max_lr * (it + 1) / warmup_iters
        if it > max_iters:
            return min_lr

        # Apply cosine decay
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)  # Safety bounds check
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

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

    epoch = start_epoch
    current_loss = float("inf")

    # Train until target loss is reached or max_epochs is hit
    with tqdm(total=max_epochs, desc="Epochs", initial=start_epoch) as epoch_pbar:
        while epoch < max_epochs and current_loss > target_loss:
            total_loss = 0
            batch_count = 0
            tokens_per_sec_avg = 0

            # Training loop for this epoch
            with tqdm(total=total_batches, desc=f"Epoch {epoch} batches") as batch_pbar:
                for i in range(total_batches):
                    t0 = time.time()

                    # Get batch
                    x, y = train_loader.next_batch()
                    x, y = x.to(device), y.to(device)

                    # Adjust learning rate
                    lr = get_lr(epoch, i, total_batches)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                    # Forward and backward
                    optimizer.zero_grad()

                    # Use mixed precision when supported
                    if device == "cuda" and hasattr(torch, "autocast"):
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            logits, loss = model(x, y)
                    else:
                        logits, loss = model(x, y)

                    loss.backward()

                    # Apply gradient clipping
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()

                    # Calculate training speed
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.time()
                    dt = t1 - t0
                    tokens_per_sec = (train_loader.B * train_loader.T) / dt
                    tokens_per_sec_avg = (
                        0.9 * tokens_per_sec_avg + 0.1 * tokens_per_sec
                        if batch_count > 0
                        else tokens_per_sec
                    )

                    total_loss += loss.item()
                    batch_count += 1

                    # Update batch progress bar
                    batch_pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{lr:.1e}",
                        tok_per_sec=f"{tokens_per_sec:.0f}",
                    )
                    batch_pbar.update(1)

            # Calculate average loss for the epoch
            current_loss = total_loss / batch_count

            # Update epoch progress bar
            epoch_pbar.set_postfix(
                loss=f"{current_loss:.4f}",
                best=f"{best_loss:.4f}",
                tok_per_sec=f"{tokens_per_sec_avg:.0f}",
            )
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
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
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
        "--weight_decay", type=float, default=0.1, help="Weight decay coefficient"
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

        # Add configure_optimizers method to model
        if not hasattr(model, "configure_optimizers"):
            model.configure_optimizers = GPT.configure_optimizers.__get__(model)

        # Initialize data loader with larger batch size and context length
        train_loader = DataLoaderLite(B=args.batch_size, T=args.seq_length)

        # Train the model with improved training
        train(
            model,
            train_loader,
            args.lr,
            args.target_loss,
            args.max_epochs,
            args.weight_decay,
            args.resume,
        )

    elif args.mode == "generate":
        # Initialize model
        model = GPT(GPTConfig())

        # Load the best model
        model, _, _ = load_checkpoint(
            model, path="checkpoints/best", map_location=torch.device("cpu")
        )
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
