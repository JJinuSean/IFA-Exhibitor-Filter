import os
from typing import List, Optional

import litellm
import numpy as np
from tqdm import tqdm


def get_text_embedding(
    text: List[str],
    api_base: str = None,
    api_key: str = None,
    model: str = None,
    batch_size: Optional[int] = 32,
    api_version: Optional[str] = "2023-07-01-preview",
) -> List[np.ndarray]:
    if model is None:
        model = "text-embedding-3-large"
    if not model.startswith("azure/"):
        model = "azure/" + model
        
    embeddings = []

    for i in tqdm(range(0, len(text), batch_size), desc="Embedding batches"):
        batch_texts = text[i:i + batch_size]

        completion_kwargs = {
            "input": batch_texts,
            "api_base": api_base or os.environ["AZURE_OPENAI_ENDPOINT"], 
            "api_key": api_key or os.environ["AZURE_OPENAI_API_KEY"],
            "api_version": api_version,
            "model": model,
        }
    
        response = litellm.embedding(**completion_kwargs)
        batch_embeddings = [np.array(emb['embedding']) for emb in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings