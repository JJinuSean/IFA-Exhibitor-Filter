import os
from typing import List, Optional

import litellm
import numpy as np


def get_text_embedding(
    text: List[str],
    api_base: str = None,
    api_key: str = None,
    model: str = None,
    api_version: Optional[str] = "2023-07-01-preview",
) -> np.ndarray:
    if model is None:
        model = "text-embedding-3-large"
    if not model.startswith("azure/"):
        model = "azure/" + model

    completion_kwargs = {
        "input": text,
        "api_base": api_base or os.environ["AZURE_OPENAI_ENDPOINT"], 
        "api_key": api_key or os.environ["AZURE_OPENAI_API_KEY"],
        "api_version": api_version,
        "model": model,
    }
    
    response = litellm.embedding(**completion_kwargs)
    emb_list = [np.array(emb['embedding']) for emb in response.data]
    return emb_list