from flask import Flask, request, jsonify, render_template
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

# Load model and tokenizer once at startup
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def predict_next_word(text, max_length=5):
    """Generates a continuation of the input text."""
    inputs = tokenizer(text, return_tensors="pt")
    input_length = len(inputs["input_ids"][0])
    
    # Ensure max_length is reasonable
    max_length = min(input_length + max_length, 50)  

    outputs = model.generate(
        **inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id  # Avoids padding error
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle the text prediction API request."""
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        predicted_text = predict_next_word(text)
        return jsonify({"input": text, "prediction": predicted_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
