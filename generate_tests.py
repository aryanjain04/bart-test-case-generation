from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def generate_test(function_code):
    model_name = "facebook/bart-large"

    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    prompt = f"""
Generate Python unit tests using pytest for the following function:

{function_code}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=200,
            num_beams=4,
            early_stopping=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    sample_function = """
def add(a, b):
    return a + b
"""

    result = generate_test(sample_function)
    print("Generated Output:\n")
    print(result)
