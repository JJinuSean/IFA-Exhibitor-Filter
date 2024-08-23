import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm


def scores_to_csv(
    companies_info: List[Dict[str, Union[str, np.ndarray]]],
    scores: Dict[str, float],
    csv_name: Optional[str]
):
    for company_info in companies_info:
        company_name = company_info['company']
        company_info['score'] = scores.get(company_name, None)
    
    score_df = pd.DataFrame(companies_info, columns=['company', 'brand_info', 'url', 'score'])
    score_df = score_df[score_df['score'].notna()]
    score_df.to_csv(csv_name, index=False)


def get_company_info(
    companies_info: List[Dict[str, Union[str, np.ndarray]]],
    company_name: str
) -> Dict[str, Union[str, np.ndarray]]:
    company_name_normalized = company_name.replace(" ", "").lower()

    for company in companies_info:
        if company['company'].replace(" ", "").lower() == company_name_normalized:
            return company
        
    raise ValueError(f"Company '{company_name}' not found in the list of companies.")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return dot(a, b) / (norm(a) * norm(b))