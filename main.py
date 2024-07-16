import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter
import nltk
from nltk.corpus import stopwords
from langdetect import detect, DetectorFactory
import cohere
import time
import asyncio
import websockets
import json
import threading

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
DetectorFactory.seed = 0

app = Flask(__name__)
CORS(app)

# Load the sentiment analysis model and tokenizer
reviews = pd.read_csv("IMDB Dataset.csv")
sentences = reviews['review'].to_numpy()

model = keras.models.load_model("z_model")
vocab_size = 10000
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)
sequence_length = 200

cohere_api_key = 'QU02t6xKfimEXivq4VGkdXkpTWDz4MjHUFywGMjb'
cohere_client = cohere.Client(cohere_api_key)

# Supabase credentials
supabase_url = 'https://hazuyoitratmohbfmzab.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhhenV5b2l0cmF0bW9oYmZtemFiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjA1MzkyNTUsImV4cCI6MjAzNjExNTI1NX0.OPCwTg7eU2047EQeZUm-IMrOsaeORsUMtMTWsLylr4U'

def http_client():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
        }
    )

    def log_url(res, *args, **kwargs):
        logging.info(f"{res.url}, {res.status_code}")

    session.hooks["response"] = log_url
    return session

def make_request(client, url: str):
    try:
        response = client.get(url)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logging.warning(f"HTTP Error for URL {url}: {e}")
        return None

def extract_brand_from_search_item(item):
    brand_tag = item.select_one("span.a-size-medium.a-color-base")
    if brand_tag:
        return brand_tag.text.strip()
    return None

def extract_asins_from_search(response, query):
    query_brand = query.split()[0].lower()
    soup = BeautifulSoup(response.text, 'lxml')
    asins = []
    for link in soup.select("div[data-asin]"):
        asin = link.get("data-asin")
        sponsored_label = link.select_one("span.s-label-popover-default")
        brand = extract_brand_from_search_item(link)
        if asin and not sponsored_label and brand and query_brand in brand.lower():
            asins.append(asin)
    return asins

def extract_reviews(response):
    soup = BeautifulSoup(response.text, 'lxml')
    reviews = []
    for review in soup.select("div.review"):
        try:
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

def predict_sentiment(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=sequence_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequences, verbose=False)
    prediction_value = prediction[0][0]
    
    if prediction_value < 0.4:
        category = 'negative'
    elif prediction_value < 0.6:
        category = 'neutral'
    else:
        category = 'positive'
    
    return category, float(prediction_value)

def extract_keywords(reviews, max_keywords=4):
    all_words = ' '.join(reviews).lower().split()
    filtered_words = [word for word in all_words if word not in stop_words and word.isalpha()]
    word_counts = Counter(filtered_words)
    most_common_words = [word for word, _ in word_counts.most_common(max_keywords)]
    return most_common_words

def generate_combined_summary(positive_reviews, negative_reviews):
    concatenated_positive_reviews = ' '.join(positive_reviews[:1000])
    concatenated_negative_reviews = ' '.join(negative_reviews[:1000])
    combined_reviews = f"Positive reviews: {concatenated_positive_reviews} Negative reviews: {concatenated_negative_reviews}"
    
    response = cohere_client.chat(
        model='command-r',
        message=f"Generate a summary of the following product reviews, discussing both positive and negative aspects: {combined_reviews}",
        max_tokens=300
    )
    return response.text.strip()

def analyze_query(query):
    client = http_client()
    base_search_url = f"https://www.amazon.co.uk/s?k={query.replace(' ', '+')}"
    current_page = 1
    all_reviews = []
    
    while len(all_reviews) < 50:
        search_url = f"{base_search_url}&page={current_page}"
        search_response = make_request(client, search_url)
        if not search_response:
            logging.info("Search request failed.")
            return {'error': 'Search request failed'}

        asins = extract_asins_from_search(search_response, query)
        baseurl = "https://www.amazon.co.uk/dp/"
        
        for asin in asins:
            product_url = baseurl + asin
            product_response = make_request(client, product_url)
            
            if not product_response:
                logging.info(f"Skipping ASIN {asin} due to request error.")
                continue
            
            reviews = extract_reviews(product_response)
            english_reviews = [review for review in reviews if is_english(review)]
            all_reviews.extend(english_reviews[:50])
            
            if len(all_reviews) >= 50:
                break
        
        if len(all_reviews) >= 50:
            break

        current_page += 1
        time.sleep(1)

    if not all_reviews:
        return {'error': 'No reviews found'}

    sentiments = [predict_sentiment(review) for review in all_reviews]
    average_confidence = np.mean([confidence for _, confidence in sentiments])
    
    positive_reviews = [review for review, (category, _) in zip(all_reviews, sentiments) if category == 'positive']
    neutral_reviews = [review for review, (category, _) in zip(all_reviews, sentiments) if category == 'neutral']
    negative_reviews = [review for review, (category, _) in zip(all_reviews, sentiments) if category == 'negative']

    positive_words = extract_keywords(positive_reviews)
    negative_words = extract_keywords(negative_reviews)
    
    combined_summary = generate_combined_summary(positive_reviews, negative_reviews)
    
    result = {
        'total_reviews_collected': len(all_reviews),
        'average_confidence': average_confidence,
        'positive_keywords': positive_words,
        'negative_keywords': negative_words,
        'combined_summary': combined_summary,
    }
    
    return result

def update_result_in_supabase(record_id, result):
    logging.info(f'Updating result in Supabase for record ID: {record_id}')
    if not result:
        logging.error('Result is empty or null')
        return
    try:
        # Prepare data for Supabase update
        positive_keywords = ', '.join(result['positive_keywords'])
        negative_keywords = ', '.join(result['negative_keywords'])
        combined_summary = result['combined_summary']
        
        # Clean the summaries from unwanted characters like {, ", }
        combined_summary = combined_summary.replace('"', '').replace('{', '').replace('}', '')

        data = {
            'result': combined_summary,
            'positive_keywords': positive_keywords,
            'negative_keywords': negative_keywords,
            'analysis_done': True  # Set the analysis_done field to True
        }
        
        response = requests.patch(
            f'{supabase_url}/rest/v1/analyze_results?id=eq.{record_id}',
            headers={
                'apikey': supabase_key,
                'Authorization': f'Bearer {supabase_key}',
                'Content-Type': 'application/json'
            },
            json=data
        )
        
        logging.info(f'Supabase update response status code: {response.status_code}')
        logging.info(f'Supabase update response text: {response.text}')
        if response.status_code != 204:
            response.raise_for_status()
            logging.info(f'Supabase update response JSON: {response.json()}')
    except requests.exceptions.HTTPError as e:
        logging.error(f'HTTP error updating Supabase: {e}')
    except requests.exceptions.RequestException as e:
        logging.error(f'Request exception updating Supabase: {e}')
    except json.JSONDecodeError:
        logging.error(f'Error decoding JSON response from Supabase: {response.text}')

@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.form.get('query')
    if not query:
        logging.info('Query not provided')
        return jsonify({'error': 'Query not provided'}), 400

    logging.info(f'Received query: {query}')
    result = analyze_query(query)
    return jsonify(result)

async def listen_for_new_queries():
    async with websockets.connect(f"{supabase_url.replace('https', 'wss')}/realtime/v1/websocket?apikey={supabase_key}") as websocket:
        subscribe_message = {
            "type": "subscribe",
            "topic": "realtime:public:analyze_results",
            "event": "phx_join",
            "payload": {},
            "ref": "1"
        }
        await websocket.send(json.dumps(subscribe_message))

        logging.info("Websocket connection established.")
        
        try:
            while True:
                message = await websocket.recv()
                message_data = json.loads(message)
                
                logging.info(f"Websocket message received: {message_data}")
                
                if message_data.get('event') == 'INSERT':
                    new_record = message_data['payload']['record']
                    query = new_record['query']
                    record_id = new_record['id']
                    
                    logging.info(f"New query detected: {query}")
                    result = analyze_query(query)
                    logging.info(f'Analysis result: {result}')
                    update_result_in_supabase(record_id, result)
        except websockets.exceptions.ConnectionClosed:
            logging.warning("Websocket connection closed.")
        except Exception as e:
            logging.error(f"Error in websocket connection: {e}")

def start_flask():
    app.run(port=6523)

def start_websocket_listener():
    while True:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(listen_for_new_queries())
        except Exception as e:
            logging.error(f"Error in websocket listener: {e}")
            time.sleep(5)  # Reconnect after a delay

if __name__ == '__main__':
    logging.basicConfig(filename='amazonscraper.log', format='%(asctime)s %(message)s', level=logging.INFO)
    
    flask_thread = threading.Thread(target=start_flask)
    websocket_thread = threading.Thread(target=start_websocket_listener)
    
    flask_thread.start()
    websocket_thread.start()

    flask_thread.join()
    websocket_thread.join()
