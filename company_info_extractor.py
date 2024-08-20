from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from bs4 import BeautifulSoup
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

from embeddings import get_text_embedding
from translate import translate_text

_tag = (
    "body > div.site > div.content > main > div.content__main__body > "
    "div > div > div.m-exhibitor-entry__item.js-library-list.js-library-item."
    "js-library-entry-item > div.m-exhibitor-entry__item__body > div"
)


def extract_info(
    companies: List[str],
    full_urls: List[str],
    tag: str = None,
) -> Dict[str, np.ndarray]:
    tag = tag or _tag
    # extract company introduction
    information = {}
    start_txt = "Brand info:"
    for i, url in enumerate(tqdm(full_urls)):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            elements = soup.select(tag)
            if elements:
                text = elements[0].text
                idx = text.find(start_txt)
                if idx != -1:
                    text = translate_text(text[idx + len(start_txt):].strip())
                    information[companies[i]] = text
                else:
                    print(f"'{start_txt}' not found in the content from URL: {url}")
            else:
                print(f"Tag not found for URL: {url}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed for URL: {url} with error: {e}")

    # introduction embedding
    info_list = list(information.values())
    embeddings = get_text_embedding(text=info_list)
    embed_info = {key: embeddings[i] for i, key in enumerate(information.keys())}
    return embed_info


def get_top_k_similar_companies(
    embed_info: Dict[str, np.ndarray],
    input: Dict[str, np.ndarray],
    k: Optional[int] = 200
) -> list[str]:
    score = {}
    input_emb = list(input.values())[0]

    for company, info_emb in embed_info.items():
        score[company] = dot(info_emb, input_emb) / (norm(info_emb) * norm(input_emb))

    scores = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))
    
    if k < len(scores.keys()): 
        top_k_result = list(scores.keys())[:k]
        return top_k_result
    else:
        raise ValueError("'k' should be smaller than the number of existing companies.")