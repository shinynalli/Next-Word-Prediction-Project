import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_model():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

def predict_next_word(model, tokenizer, text, max_length=5):
    inputs = tokenizer(text, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    
    # Ensure max_length is reasonable
    max_length = min(input_length + max_length, 50)  # 50 prevents exceeding model limits
    
    outputs = model.generate(
        **inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id  # Avoids padding error
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model()
    
    while True:
        text = input("Enter your line: ")
        
        if text.lower() == "stop the script":
            print("Ending The Program.....")
            break
        
        try:
            predicted_text = predict_next_word(model, tokenizer, text)
            print("Predicted Text:", predicted_text)
        except Exception as e:
            print("Error:", str(e))
