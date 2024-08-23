from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm


def collect_exhibitor_urls(
    base_url: Optional[str] = "https://www.ifa-berlin.com/",
    path: Optional[str] = "exhibitors",
    searchgroup: Optional[str] = "00000001-exhibitors",
    start_page: Optional[int] = 1, 
    end_page: Optional[int] = 75
) -> Tuple[list[str], list[str]]:
    company_list = []
    ifa_companies_links = []
    for i in tqdm(range(start_page, end_page + 1), desc="Collecting Exhibitors"):
        url = f"{base_url}{path}?&page={i}&searchgroup={searchgroup}"
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.content
        else:
            raise Exception(f"Failed to retrieve content: {response.status_code}")
        soup = BeautifulSoup(html_content, 'html.parser')
        companies_links = soup.select('div div main div:nth-of-type(2) div div div div ul li div:nth-of-type(2) a')
        ifa_companies_links.extend(
            [link.get("href") for link in companies_links if link.get("href").startswith(path)]
        )
        company_list.extend(
            [link.get('aria-label') for link in companies_links if link.get('aria-label') is not None]
        )
    full_urls = [
        base_url + postfix for postfix in ifa_companies_links
    ]
    
    return company_list, full_urls