import logging
import os
import openai
import traceback

logger = logging.getLogger("ai_service")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
global_rag = {"value": "gpt-4o"}

def generate_text_with_model(input_text, model_name=None, temperature=0.5):
    """
    Sends a plain text prompt to the configured OpenAI model and returns the output text.

    Args:
        input_text (str): The text prompt or question to send to the model.
        model_name (str): Optional. Model to use (default: global_rag["value"] if defined).
        temperature (float): Optional. Sampling temperature for creativity (0.0–1.0).

    Returns:
        str: Model-generated response text, or an error message.
    """

    # Choose default model if not passed
    try:
        model_to_use = model_name or global_rag["value"]
    except NameError:
        model_to_use = "gpt-4o-mini"  # safe fallback

    print("\n====================== 🧠 MODEL INPUT DEBUG ======================")
    print(input_text)
    print("=================================================================\n")

    try:
        response = openai.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": input_text}],
            temperature=temperature,
        )

        output_text = response.choices[0].message.content.strip()

        print("\n====================== 🤖 MODEL OUTPUT ======================")
        print(output_text)
        print("================================================================\n")

        return output_text

    except Exception as e:
        print(f"⚠️ Error during model call: {str(e)}")
        traceback.print_exc()
        return f"⚠️ Error: {str(e)}"