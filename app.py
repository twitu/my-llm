import os
import torch
import gradio as gr
from model import GPT, GPTConfig
import torch.nn.functional as F
from train import load_checkpoint
import time

# Set device to CPU for Hugging Face deployment
device = torch.device("cpu")


def load_tokenizer():
    """Load the tokenizer"""
    # Assuming you're using a standard tokenizer like GPT-2's
    try:
        from transformers import GPT2Tokenizer

        return GPT2Tokenizer.from_pretrained("gpt2")
    except:
        # Fallback to a simple character-level tokenizer if transformers isn't available
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) for c in text]

            def decode(self, tokens):
                return "".join([chr(t) for t in tokens])

        return SimpleTokenizer()


# Initialize model and load checkpoint
def initialize_model():
    model = GPT(GPTConfig())
    model, _, _ = load_checkpoint(model, map_location=device)
    model.eval()  # Set to evaluation mode
    return model


# Load tokenizer
tokenizer = load_tokenizer()


# Generate text based on prompt
def generate_text(
    prompt, max_new_tokens=50, temperature=0.8, top_k=50, num_return_sequences=1
):
    model = initialize_model()

    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids] * num_return_sequences, dtype=torch.long).to(device)

    # Generate text
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits from the model
            logits = model(x)[0]  # Assuming model returns (logits, loss) or just logits

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

    # Decode the generated sequences
    generated_texts = []
    for i in range(num_return_sequences):
        tokens = x[i].tolist()
        decoded = tokenizer.decode(tokens)
        generated_texts.append(decoded)

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
