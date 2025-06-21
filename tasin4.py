Python 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import torch
... from transformers import T5Tokenizer, T5ForConditionalGeneration
... 
... # Load pre-trained T5 model and tokenizer
... model = T5ForConditionalGeneration.from_pretrained('t5-base')
... tokenizer = T5Tokenizer.from_pretrained('t5-base')
... 
... def generate_text(topic, max_length=200):
...     # Define the prompt
...     prompt = f"Generate a paragraph about {topic}."
... 
...     # Encode the prompt
...     input_ids = tokenizer.encode(prompt, return_tensors='pt')
... 
...     # Generate text
...     output = model.generate(input_ids, max_length=max_length)
... 
...     # Decode the generated text
...     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
... 
...     return generated_text
... 
... # Test the function
... topic = "Artificial Intelligence"
... generated_text = generate_text(topic)
