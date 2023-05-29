import json
import requests
import os
import openai
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s",
    handlers=[logging.FileHandler("llm.log"), logging.StreamHandler()],
)


def call_gpt(model, context, prompt):
    try:
        system_prompt = context
        completion = openai.ChatCompletion.create(
            model=model, 
            messages=[{"role": "system", "content" : system_prompt }, {"role": "user", "content": prompt}]
        )
        return completion['choices'][0]["message"]["content"]
    except Exception as e:
        logging.error("GPT Function failed! " + str(e))
        return None
