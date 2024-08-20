from typing import Dict, List

import numpy as np
import requests
from bs4 import BeautifulSoup
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

from embeddings import get_text_embedding
from translate import translate_text


class CompanyInfoExtractor:
    def __init__(
        self,
        start_txt="Brand info:",
        tag=(
            "body > div.site > div.content > main > div.content__main__body > "
            "div > div > div.m-exhibitor-entry__item.js-library-list.js-library-item."
            "js-library-entry-item > div.m-exhibitor-entry__item__body > div"
        )
    ):
        self.start_txt = start_txt
        self.tag = tag
        self.information = {}
        self.embed_info = None

    def extract_info(self, companies, full_urls) -> None:
        # extract company introduction
        for i, url in enumerate(tqdm(full_urls)):
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                elements = soup.select(self.tag)
                if elements:
                    text = elements[0].text
                    idx = text.find(self.start_txt)
                    if idx != -1:
                        text = translate_text(text[idx + len(self.start_txt) :].strip())
                        self.information[companies[i]] = text
                    else:
                        print(
                            f"'{self.start_txt}' not found in the content from URL: {url}"
                        )
                else:
                    print(f"Tag not found for URL: {url}")
            except requests.exceptions.RequestException as e:
                print(f"Request failed for URL: {url} with error: {e}")
            if i == 3: break
        
        # introduction embedding
        info_list = list(self.information.values())
        embeddings = get_text_embedding(text=info_list)
        self.embed_info = {
            key: embeddings[i] for i, key in enumerate(self.information.keys())
        }
    
    def get_top_k_similar_companies(self, input, k) -> dict[str, int]:
        score = {}
        input_emb = list(input.values())[0]
           
        for company, info_emb in self.embed_info.items():
            score[company] = dot(info_emb, input_emb) / (norm(info_emb)*norm(input_emb))

        scores = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))
        return list(scores.keys())[:k]
        