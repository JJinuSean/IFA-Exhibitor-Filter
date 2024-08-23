from typing import Dict, List, Optional, Union

import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from embeddings import get_text_embedding
from translation import translate_text
from utils import cosine_similarity

_tag = (
    "body > div.site > div.content > main > div.content__main__body > "
    "div > div > div.m-exhibitor-entry__item.js-library-list.js-library-item."
    "js-library-entry-item > div.m-exhibitor-entry__item__body > div"
)


def fetch_company_info(url: str, tag: str, start_txt: str) -> Optional[str]:
    """Fetches and returns the company information from the given URL."""
    try:
        response = requests.get(url)    
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        elements = soup.select(tag)
        if elements:
            text = elements[0].text
            idx = text.find(start_txt)
            if idx != -1:
                return text[idx + len(start_txt) :].strip()
        else:
            print(f"Tag not found for URL: {url}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for URL: {url} with error: {e}")
        
    return None


def extract_info(
    companies: List[str],
    full_urls: List[str],
    tag: Optional[str] = None,
    batch_size: Optional[int] = 32,
) -> List[Dict[str, Union[str, np.ndarray]]]:
    tag = tag or _tag
    start_txt = "Brand info:"
    information = []

    for i, url in enumerate(tqdm(full_urls, desc='Collecting brand info')):
        brand_info = fetch_company_info(url, tag, start_txt)
        if brand_info:
            information.append(
                {"company": companies[i], "brand_info": brand_info, "url": url}
            )
    texts = [info["brand_info"] for info in information]
    translated_texts = translate_text(texts, batch_size=batch_size)
    embeddings = get_text_embedding(translated_texts, batch_size=batch_size)

    for i in range(len(information)):
        information[i]["brand_info"] = translated_texts[i]
        information[i]["embedding"] = embeddings[i]

    return information


def get_top_k_similar_companies(
    companies_info: List[Dict[str, Union[str,np.ndarray]]],
    my_company_info: Dict[str, Union[str,np.ndarray]],
    k: Optional[int] = 5,
) -> Dict[str, float]:
    companies_emb = [company['embedding'] for company in companies_info]
    my_company_emb = my_company_info['embedding']

    scores = {}
    for i, company_emb in enumerate(companies_emb):
        score = cosine_similarity(my_company_emb, company_emb)
        scores[companies_info[i]['company']] = score

    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    
    if k < len(sorted_scores):
        return dict(list(sorted_scores.items())[1:k+1])
    else:
        raise ValueError("'k' should be smaller than the number of existing companies.")