import argparse
import pandas as pd

from collect_exhibitor_urls import collect_exhibitor_urls
from company_info_extractor_refactor import extract_info, get_top_k_similar_companies
from utils import scores_to_csv, get_company_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--company_name", type=str, required=True)
    parser.add_argument("--result", type=str, required=False, default="result.csv")
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--top_k", type=int, required=False, default=5)

    args = parser.parse_args()
    
    companies, urls = collect_exhibitor_urls()
    companies_info = extract_info(companies, urls, batch_size=args.batch_size)
    my_company_info = get_company_info(companies_info, args.company_name)

    scores = get_top_k_similar_companies(companies_info, my_company_info, k=args.top_k)
    scores_to_csv(companies_info, scores, args.result)


if __name__ == "__main__":
    main()