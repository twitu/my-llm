---
title: My Llm
emoji: 🌖
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
short_description: My own shakespeare LLM
---

# GPT Text Generation App

This is a simple text generation app using a custom GPT model. Enter a prompt, adjust the generation parameters, and see what the model creates!

## Parameters

- **Maximum New Tokens**: Controls how many new tokens the model will generate.
- **Temperature**: Controls randomness. Higher values (e.g., 1.5) make output more random, lower values (e.g., 0.2) make it more focused and deterministic.
- **Top-k**: Limits the next token selection to the k most likely tokens.
- **Number of Sequences**: Generate multiple different outputs for the same prompt.

## Model

This app uses a custom GPT model trained on [describe your dataset here]. 
