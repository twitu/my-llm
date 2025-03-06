import os
import torch
import gradio as gr
from model import GPT, GPTConfig
from train import CPU_Unpickler
import torch.nn.functional as F
import time
from huggingface_hub import hf_hub_download

# Set device to CPU for Hugging Face deployment
device = torch.device("cpu")

# Hugging Face model repository information
HF_USERNAME = "twitu"  # Replace with your actual username
HF_MODEL_REPO = "my-llm-model"
CHECKPOINT_FILENAME = "latest_checkpoint.pkl"


def download_model_from_hub():
    """Download the model checkpoint from Hugging Face Hub"""
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", CHECKPOINT_FILENAME)

    # Skip download if file already exists
    if os.path.exists(checkpoint_path):
        print(f"Using existing checkpoint at {checkpoint_path}")
        return checkpoint_path

    print(f"Downloading model from {HF_USERNAME}/{HF_MODEL_REPO}...")
    try:
        # Download the checkpoint file from Hugging Face Hub
        downloaded_path = hf_hub_download(
            repo_id=f"{HF_USERNAME}/{HF_MODEL_REPO}",
            filename=CHECKPOINT_FILENAME,
            cache_dir="checkpoints",
        )
        print(f"Model downloaded successfully to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None


def load_checkpoint(model, map_location=None):
    """Load model checkpoint from file"""
    # Download or get the checkpoint path
    checkpoint_path = download_model_from_hub()

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None, 0, float("inf")

    # Use torch.load with map_location parameter
    checkpoint = CPU_Unpickler(open(checkpoint_path, "rb")).load()

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint loaded from {checkpoint_path}")
    return model, checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))


def load_tokenizer():
    """Load the tokenizer"""
    # Assuming you're using a standard tokenizer like GPT-2's
    try:
        from transformers import GPT2Tokenizer

        return GPT2Tokenizer.from_pretrained("gpt2")
    except:
        # Fallback to a simple character-level tokenizer if transformers isn't available
        import tiktoken

        return tiktoken.get_encoding("gpt2")


# Initialize model and load checkpoint
def initialize_model():
    model = GPT(GPTConfig())
    model, _, _ = load_checkpoint(model, map_location=device)
    model.eval()  # Set to evaluation mode
    return model


# Load tokenizer
tokenizer = load_tokenizer()
model = initialize_model()


# Generate text based on prompt with streaming
def generate_text(
    prompt, max_new_tokens=50, temperature=0.8, top_k=50, num_return_sequences=1
):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids] * num_return_sequences, dtype=torch.long).to(device)

    # Initialize the output text with the prompt
    generated_texts = [prompt] * num_return_sequences

    # Generate text token by token
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits from the model
            logits, _ = model(x)  # Assuming model returns (logits, loss)

            # Take the logits at the last position
            logits = logits[:, -1, :] / temperature  # Apply temperature

            # Apply top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

            # Apply softmax to get probabilities
            probs = F.softmax(top_k_logits, dim=-1)

            # Sample from the distribution
            next_token_idx = torch.multinomial(probs, 1)

            # Get the actual token ids
            next_tokens = torch.gather(top_k_indices, -1, next_token_idx)

            # Append to the sequence
            x = torch.cat((x, next_tokens), dim=1)

            # Decode and update the generated text
            for i in range(num_return_sequences):
                token = next_tokens[i].item()
                token_text = tokenizer.decode([token])
                generated_texts[i] += token_text

            # Yield the current state of all generated texts
            yield "\n\n---\n\n".join(generated_texts)

            # Add a small delay to make the streaming visible
            time.sleep(0.05)

    return "\n\n---\n\n".join(generated_texts)


# Create Gradio interface
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Text Generation with GPT Model")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt", placeholder="Enter your prompt here...", lines=5
                )
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                    label="Maximum New Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"
                )
                top_k = gr.Slider(
                    minimum=1, maximum=100, value=50, step=1, label="Top-k"
                )
                num_sequences = gr.Slider(
                    minimum=1, maximum=5, value=1, step=1, label="Number of Sequences"
                )
                generate_btn = gr.Button("Generate")

            with gr.Column():
                output = gr.Textbox(label="Generated Text", lines=10)

        generate_btn.click(
            fn=generate_text,
            inputs=[prompt, max_tokens, temperature, top_k, num_sequences],
            outputs=output,
        )

    return demo


# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
