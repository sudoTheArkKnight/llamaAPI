from flask import Flask, request, jsonify
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# Create a Flask application instance
app = Flask(__name__)

# Specify the model name as per the Hugging Face repository or local path
model_name = "meta-llama/Llama-3.2-1B"

# Load the tokenizer and model using the pre-trained LLaMA model
# Ensure the correct model path or identifier is provided,
# and it should be accessible either locally or from Hugging Face
# The model is loaded on the CPU by default as it is more accessible without a GPU
# Note: Loading such large models can be memory intensive
# Tokenizer converts strings into token IDs that the model can understand
try:
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

# Define a route for text generation
@app.route('/generate', methods=['POST'])
def generate():
    # Get JSON data from the request and extract the prompt provided by the user
    data = request.json
    prompt = data.get('prompt', '')

    # If no prompt is provided, return an error message to the user
    if not prompt:
        return jsonify({"error": "Please provide a prompt"}), 400

    # Tokenize the input prompt to convert it to a format suitable for model processing
    # The tokenizer returns token IDs as PyTorch tensors
    inputs = tokenizer(prompt, return_tensors="pt").to('cpu')

    # Generate text based on the input prompt without calculating gradients (to save resources)
    # Set 'max_length' to control the output length and 'num_return_sequences' to specify the number of responses
    try:
        with torch.no_grad():  # Disable gradient computation for faster inference since we are not training the model
            outputs = model.generate(
                inputs['input_ids'],  # Tokenized input prompt as tensor
                max_length=100,       # Maximum length of the output sequence
                num_return_sequences=1 # Number of response sequences to generate
            )
    except Exception as e:
        # Handle any errors that might occur during text generation
        return jsonify({"error": f"Failed to generate text: {e}"}), 500

    # Decode the generated token IDs back into text using the tokenizer
    # Skip special tokens to produce more readable text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the generated text as a JSON response
    return jsonify({'generated_text': generated_text})

# Start the Flask server when the script is executed directly
# The server will be available locally on port 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# Sample Input inluding command: curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d '{"prompt": "Explain a database in less than 50 words."}'
# Sample Output: {"generated_text":"Explain Databases in less 50 words\nA database is a collection of organized data, typically stored in a computer. It helps to manage, store, and retrieve data efficiently. Databases use a structured format, with each piece of data linked to a unique identifier, making it easy to search, update, and maintain. Examples include relational databases like MySQL and MongoDB."}