#!/bin/bash

# Step 1: Download the model file to the current folder
echo "Downloading Mistral-7B-Instruct model..."
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf -O mistral-7b-instruct-v0.2.Q5_K_M.gguf

echo "Download complete!"