import os
from typing import Dict, List, Optional

import litellm
from tqdm import tqdm


def translate_text(
    texts: List[str],
    api_base: str = None,
    api_key: str = None,
    model: str = None,
    batch_size: int = 32,
    api_version: Optional[str] = "2023-07-01-preview"
) -> List[str]:
    if model is None:
        model = "gpt-35-turbo-16k-0613"
    if not model.startswith("azure/"):
        model = "azure/" + model

    prompt_template = """
    You are a translator.
    Your task is to translate the given '# source text' into English.

    # source text
    {text}

    # translated result
    """.lstrip()

    translated_texts = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Translating batches"):
        batch_texts = texts[i:i + batch_size]
        
        jointed_texts = "\n\n".join([f"<SEP{i}> {text}" for i, text in enumerate(batch_texts)])
        jointed_texts += f"<SEP{len(batch_texts)}>"

        prompt = prompt_template.format(text=jointed_texts)
        messages = [
            {"role": "user", "content": prompt},
        ]

        completion_kwargs = {
            "messages": messages,
            "api_base": api_base or os.environ['AZURE_OPENAI_ENDPOINT'],
            "api_key": api_key or os.environ['AZURE_OPENAI_API_KEY'],
            "api_version": api_version,
            "model": model,
        }
        response = litellm.completion(**completion_kwargs)
        translated_batch = response.choices[0].message.content.strip()

        for j in range(len(batch_texts)):
            start_tag = f"<SEP{j}>"
            end_tag = f"<SEP{j+1}>" if j < len(batch_texts) else None

            start_idx = translated_batch.find(start_tag)
            end_idx = translated_batch.find(end_tag)

            translated_text = translated_batch[start_idx + len(start_tag):end_idx].strip()
            translated_texts.append(translated_text)

    return translated_texts