import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import time
from langdetect import detect, DetectorFactory
from tqdm import tqdm

DetectorFactory.seed = 0

def http_client():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
        }
    )
    return session

def make_request(client, url: str):
    try:
        response = client.get(url)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logging.warning(f"HTTP Error for URL {url}: {e}")
        return None

def extract_brand_from_product_page(soup):
    try:
        brand_tag = soup.find("tr", {"id": "bylineInfo"}).find("a").text.strip()
        return brand_tag
    except AttributeError:
        return None

def extract_product_type_from_query(query):
    return query.split()[0]  # Extract the product type from the query

def extract_asins_from_search(response):
    soup = BeautifulSoup(response.text, 'lxml')
    asins = []
    for link in soup.select("div[data-asin]"):
        asin = link.get("data-asin")
        sponsored_label = link.select_one("span.s-label-popover-default")
        if asin and not sponsored_label:
            asins.append(asin)
    return asins

def extract_reviews(response):
    soup = BeautifulSoup(response.text, 'lxml')
    reviews = []
    for review in soup.select("div.review"):
        try:
            rating = review.select_one("i.review-rating span").text
            if "1.0 out of 5 stars" in rating or "2.0 out of 5 stars" in rating:
                review_text = review.select_one("span.review-text").text.strip()
                reviews.append(review_text)
        except AttributeError:
            continue
    return reviews

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def scrape_amazon_reviews(queries, max_reviews_per_product=500, max_pages_per_query=20):
    client = http_client()
    all_reviews = []
    all_brands = []
    all_product_types = []
    all_sentiments = []

    for query in queries:
        base_search_url = f"https://www.amazon.co.uk/s?k={query.replace(' ', '+')}"
        current_page = 1
        no_review_page_count = 0  # Counter for pages with no reviews
        total_reviews_for_query = 0

        while total_reviews_for_query < max_reviews_per_product and current_page <= max_pages_per_query:
            search_url = f"{base_search_url}&page={current_page}"
            search_response = make_request(client, search_url)
            if not search_response:
                logging.info("Search request failed.")
                break

            asins = extract_asins_from_search(search_response)
            if not asins:
                no_review_page_count += 1
                if no_review_page_count >= 5:
                    logging.info(f"No products found for 5 consecutive pages for query '{query}', moving to next query.")
                    break
                current_page += 1
                continue

            baseurl = "https://www.amazon.co.uk/dp/"

            page_has_reviews = False

            for asin in tqdm(asins, desc=f"Processing ASINs for query '{query}' on page {current_page}"):
                product_url = baseurl + asin
                product_response = make_request(client, product_url)

                if not product_response:
                    logging.info(f"Skipping ASIN {asin} due to request error.")
                    continue

                reviews = extract_reviews(product_response)
                english_reviews = [review for review in reviews if review and is_english(review)]
                if english_reviews:
                    page_has_reviews = True
                    no_review_page_count = 0  # Reset counter if reviews are found

                review_count_to_add = min(100, max_reviews_per_product - total_reviews_for_query)
                reviews_to_add = english_reviews[:review_count_to_add]

                all_reviews.extend(reviews_to_add)
                total_reviews_for_query += len(reviews_to_add)

                # Extract brand and product type
                soup = BeautifulSoup(product_response.text, 'lxml')
                brand = extract_brand_from_product_page(soup)
                product_type = extract_product_type_from_query(query)

                all_brands.extend([brand] * len(reviews_to_add))
                all_product_types.extend([product_type] * len(reviews_to_add))
                all_sentiments.extend([""] * len(reviews_to_add))  # Placeholder for sentiment

                if total_reviews_for_query >= max_reviews_per_product:
                    break

            if not page_has_reviews:
                no_review_page_count += 1
                if no_review_page_count >= 5:
                    logging.info(f"No reviews found for 5 pages for query '{query}', moving to next query.")
                    break

            current_page += 1
            time.sleep(2)  # Add a delay to avoid overwhelming the server

    df = pd.DataFrame({
        'review': all_reviews,
        'sentiment': all_sentiments,  # Placeholder for sentiment
        'brand': all_brands,
        'product_type': all_product_types
    })

    df.to_csv('amazon_reviewsneg.csv', index=False)
    print(f"Saved {len(all_reviews)} reviews to amazon_reviews.csv")

if __name__ == "__main__":
    logging.basicConfig(filename='scraper.log', format='%(asctime)s %(message)s', level=logging.INFO)
    queries = [
        "laptop Dell", "laptop HP", "laptop Lenovo", "laptop Asus", "laptop Acer",
        "smartphone Samsung", "smartphone Apple", "smartphone Google", "smartphone OnePlus", "smartphone Xiaomi",
        "headphones Sony", "headphones Bose", "headphones Sennheiser", "headphones JBL", "headphones Beats",
        "camera Canon", "camera Nikon", "camera Sony", "camera Fujifilm", "camera Panasonic",
        "smartwatch Apple", "smartwatch Samsung", "smartwatch Garmin", "smartwatch Fitbit", "smartwatch Fossil",
        "tablet Apple", "tablet Samsung", "tablet Microsoft", "tablet Lenovo", "tablet Huawei",
        "television Samsung", "television LG", "television Sony", "television Philips", "television TCL",
        "gaming console PlayStation", "gaming console Xbox", "gaming console Nintendo", "gaming PC", "gaming laptop",
        "e-reader Kindle", "e-reader Kobo", "e-reader Nook", "e-reader PocketBook", "e-reader Onyx",
        "smart speaker Amazon Echo", "smart speaker Google Nest", "smart speaker Apple HomePod", "smart speaker Sonos", "smart speaker Bose"
    ]
    scrape_amazon_reviews(queries, max_reviews_per_product=500, max_pages_per_query=20)  # Adjusted to collect up to 500 reviews per product
