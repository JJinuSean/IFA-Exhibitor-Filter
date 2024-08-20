import os
from typing import Dict, Optional

import litellm


def translate_text(
    text: str,
    api_base: str = None,
    api_key: str = None,
    model: str = None,
    api_version: Optional[str] = "2023-07-01-preview"
) -> dict[str, str]:
    result = []
    company_info = {}

    if model is None:
        model = "gpt-35-turbo-16k-0613"
    if not model.startswith("azure/"):
        model = "azure/" + model

    prompt = """
    You are a translator.
    Your task is to translate the given '# source text' into English.

    # source text
    {text}

    # translated result
    """.lstrip()

    messages = [
        {"role": "user", "content": prompt.format(text=text)}, 
    ]

    completion_kwargs = {
        "messages": messages, 
        "api_base": api_base or os.environ['AZURE_OPENAI_ENDPOINT'], 
        "api_key": api_key or os.environ['AZURE_OPENAI_API_KEY'],
        "api_version": api_version,
        "model": model,
    }
    
    response = litellm.completion(**completion_kwargs)
    translated_text = response.choices[0].message.content

    return translated_text