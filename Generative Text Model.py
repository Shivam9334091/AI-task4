"""
Generative Text Model
---------------------
A Python script that generates coherent text based on user prompts.

Author: Your Name
Internship: ELiteTEch Intern
Task: 4 (Generative Text Model)
"""

from transformers import pipeline


def generate_text(prompt: str, max_length: int = 150, num_return_sequences: int = 1) -> list:
    """
    Generate text based on a user prompt using GPT-2.

    Args:
        prompt (str): The input text prompt.
        max_length (int): Maximum length of generated text.
        num_return_sequences (int): Number of outputs to return.

    Returns:
        list: Generated text sequences.
    """
    generator = pipeline("text-generation", model="gpt2")
    outputs = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return [output['generated_text'] for output in outputs]


if __name__ == "__main__":
    print("✨ Generative Text Model (Task 4)")
    user_prompt = input("Enter a prompt: ")

    results = generate_text(user_prompt, max_length=120, num_return_sequences=2)

    print("\n===== Generated Texts =====\n")
    for i, text in enumerate(results, 1):
        print(f"[Output {i}]:\n{text}\n")
