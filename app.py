from flask import Flask, redirect, url_for, render_template, request, jsonify
import requests
import re
import os
import math
import pandas as pd
from bs4 import BeautifulSoup
import time as t
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

nltk.download('stopwords')
stop_words = stopwords.words('indonesian')
nltk.download('punkt_tab')

vectorizer_jenis = joblib.load('./tfidf_vectorizer_jenis.pkl')
model_jenis = joblib.load('./model_xgboost_jenis.pkl',)
vectorizer_dampak = joblib.load('./tfidf_vectorizer_dampak.pkl')
model_dampak = joblib.load('./model_xgboost_dampak.pkl',)
tokenizer = AutoTokenizer.from_pretrained('./model ner')
model = AutoModelForTokenClassification.from_pretrained('./model ner')

label_mapping = {
    0: 'Banjir',
    1: 'Bencana Hidrometerologi Ekstrem',
    2: 'Gempa Bumi',
    3: 'Gunung Meletus',
    4: 'Puting Beliung',
    5: 'Tanah Longsor',
    6: 'Tsunami'
}

# Fungsi untuk membersihkan teks pada konten
def clean_text(text):
    """
    Membersihkan teks pada kolom Content:
    - Mengubah ke huruf kecil
    - Menghapus angka
    - Menghapus stopwords
    - Menghapus spasi berlebih
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_decimal_points(text):
    """
    Mengganti titik dalam angka desimal dengan placeholder <DECIMAL>.
    Contoh: '20.46' menjadi '20<DECIMAL>46'
    """
    text = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', text)
    return text

def preprocess_quoted_dots(text):
    """
    Mengganti titik dalam kalimat kutipan dengan placeholder <QUOTE_DOT>.
    Contoh: '"Informasi ini penting."' menjadi '"Informasi ini penting<QUOTE_DOT>"'
    """
    text = re.sub(r'\.(?=\")', r'<QUOTE_DOT>', text)
    return text

def preprocess_special_cases(text):
    """
    Menangani kasus khusus lainnya, seperti titik setelah singkatan dalam tanda kurung.
    Contoh: 'Senin (1/1).' menjadi 'Senin (1/1)'
    """
    text = re.sub(r'\((\d+)/(\d+)\)\.', r'(\1/\2)', text)
    return text

def preprocess_text(text):
    """
    Melakukan semua langkah preprocessing pada teks:
    - Mengganti titik dalam angka desimal dengan placeholder <DECIMAL>.
    - Mengganti titik dalam kalimat kutipan dengan placeholder <QUOTE_DOT>.
    - Menangani kasus khusus lainnya seperti titik setelah singkatan dalam tanda kurung.
    """
    text = preprocess_decimal_points(text)
    text = preprocess_quoted_dots(text)
    text = preprocess_special_cases(text)
    return text

def postprocess_decimal_points(sentences):
    """
    Mengembalikan placeholder <DECIMAL> dan <QUOTE_DOT> menjadi titik.
    """
    if isinstance(sentences, list):
        sentences = [sentence.replace('<DECIMAL>', '.') for sentence in sentences]
        sentences = [sentence.replace('<QUOTE_DOT>', '.') for sentence in sentences]
    elif isinstance(sentences, str):
        sentences = sentences.replace('<DECIMAL>', '.').replace('<QUOTE_DOT>', '.')
    else:
        raise ValueError("Input harus berupa string atau list of strings.")
    return sentences

def sentence_tokenize(text):
    """
    Tokenize kalimat menggunakan NLTK, menangani titik dalam angka desimal dan kalimat kutipan:
    - Melakukan preprocessing untuk menangani titik dalam angka desimal dan tanda baca dalam kutipan.
    - Menggunakan NLTK untuk memisahkan teks menjadi kalimat.
    - Mengembalikan titik dalam angka desimal dan tanda baca dalam kutipan setelah tokenisasi.
    """
    text = preprocess_text(text)  # Preprocessing
    sentences = sent_tokenize(text)  # Tokenisasi menggunakan NLTK
    sentences = postprocess_decimal_points(sentences)  # Postprocessing
    return sentences

# Fungsi pembantu untuk mengupdate file gabungan
def update_gabungan(new_articles, gabungan_file="data_gabungan.xlsx"):
    """
    Membaca file gabungan yang sudah ada (jika ada), menggabungkan dengan artikel baru,
    menghilangkan duplikasi berdasarkan URL, dan menyimpan kembali ke file gabungan.
    """
    if os.path.exists(gabungan_file):
        df_existing = pd.read_excel(gabungan_file)
    else:
        df_existing = pd.DataFrame(columns=["URL", "Date", "Title", "Content"])
        
    df_new = pd.DataFrame(new_articles)
    df_combined = pd.concat([df_new, df_existing], ignore_index=True)
    df_combined.drop_duplicates(subset=["URL"], keep="first", inplace=True)
    df_combined.to_excel(gabungan_file, index=False)
    print(f"Gabungan file ({gabungan_file}) telah diperbarui dengan {len(df_new)} artikel baru.")

def scrape_detik(file_name="data_detik.xlsx", gabungan_file="data_gabungan.xlsx"):
    """
    Scrapes articles from Detik (page 1) that match disaster-related keywords.
    Saves only new articles into an Excel file while keeping old ones, 
    and also updates the combined file.
    """
    base_url = "https://www.detik.com/tag/bencana-alam/?sortby=time&page=1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    keywords = [
        r"\berupsi\b", r"\bgunung meletus\b", r"\btanah longsor\b", r"\blongsor\b",
        r"\bbanjir\b", r"\bbanjir bandang\b", r"\btsunami\b",
        r"\bgempa\b", r"\bgempa bumi\b", r"\bbadai\b", r"\bputing beliung\b", r"\bangin kencang\b",
        r"\bkekeringan\b", r"\bkemarau panjang\b", r"\bhujan es\b", r"\bgelombang panas\b",
        r"\bcuaca ekstrem\b", r"\bgelombang ekstrem\b", r"\bgemuruh laut\b",
        r"\bkebakaran hutan\b", r"\bkebakaran lahan\b", r"\bkarhutla\b", r"\bapi\b"
    ]
    keyword_pattern = re.compile("|".join(keywords), re.IGNORECASE)
    
    print("\nScraping Detik...")

    # Baca file Detik yang sudah ada untuk cek URL duplikat
    if os.path.exists(file_name):
        df_existing = pd.read_excel(file_name)
        existing_urls = set(df_existing["URL"].tolist())
    else:
        df_existing = pd.DataFrame(columns=["URL", "Date", "Title", "Content"])
        existing_urls = set()

    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("article")
    if not articles:
        print("No articles found in Detik.")
        return

    new_articles = []
    for article in articles:
        link_tag = article.find("a", href=True)
        link = link_tag['href'] if link_tag else None
        if not link or "foto-news" in link or "/foto/" in link or link in existing_urls:
            continue

        title_tag = article.find("h2", class_="title")
        title = title_tag.text.strip() if title_tag else "No title"
        print(f"Processing: {title} (Detik)")

        try:
            article_response = requests.get(link, headers=headers)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(article_response.text, "html.parser")

            # Skip jika artikel merupakan "Video News"
            video_news_tag = article_soup.find("h2", class_="detail__subtitle", string="Video News")
            if video_news_tag:
                continue

            # Hapus elemen-elemen yang tidak diperlukan
            for tag in article_soup.find_all("div", class_="parallaxindetail scrollpage"):
                tag.decompose()
            for tag in article_soup.find_all("span", class_="para_caption", string="ADVERTISEMENT"):
                tag.decompose()
            for tag in article_soup.find_all("p", class_="para_caption", string="SCROLL TO CONTINUE WITH CONTENT"):
                tag.decompose()

            # Ambil tanggal
            date_tag = article_soup.find("div", class_="detail__date")
            date = date_tag.get_text(strip=True) if date_tag else "Unknown Date"
            bulan_mapping = {
                "Jan": "Jan", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr", "Mei": "May",
                "Jun": "Jun", "Jul": "Jul", "Agu": "Aug", "Sep": "Sep", "Okt": "Oct",
                "Nov": "Nov", "Des": "Dec"
            }
            match = re.search(r'(\d{1,2}) (\w{3}) (\d{4})', date)
            if match:
                day, month, year = match.groups()
                month = bulan_mapping.get(month, month)
                date_obj = datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
                date = date_obj.strftime("%d/%m/%Y")

            # Ambil konten
            paragraphs = article_soup.find_all("p")
            content = ""
            for p in paragraphs:
                p_text = ""
                previous_text = ""
                for element in p.children:
                    if element.name == "a":
                        if previous_text:
                            p_text += " "
                        p_text += element.get_text(strip=True)
                        p_text += " "
                    elif element.name is None:
                        text = element.strip()
                        if previous_text:
                            p_text += " " + text
                        else:
                            p_text += text
                    previous_text = p_text.strip()
                content += p_text.strip() + " "
            content = re.sub(r'\s+', ' ', content).strip()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching article: {link} (Detik)")
            content = "Failed to fetch content."

        if keyword_pattern.search(title) or keyword_pattern.search(content):
            new_articles.append({
                "URL": link,
                "Date": date,
                "Title": title,
                "Content": content
            })

    if new_articles:
        print(f"Found {len(new_articles)} new articles in Detik. Updating files...")
        # Update file khusus Detik
        df_new = pd.DataFrame(new_articles)
        df_detik_combined = pd.concat([df_new, df_existing], ignore_index=True)
        df_detik_combined.drop_duplicates(subset=["URL"], keep="first", inplace=True)
        df_detik_combined.to_excel(file_name, index=False)
        print(f"File {file_name} updated with {len(new_articles)} new articles from Detik.")
        # Update file gabungan
        update_gabungan(new_articles, gabungan_file)
    else:
        print("No new articles found from Detik.")

def scrape_kompas(file_name="data_kompas.xlsx", gabungan_file="data_gabungan.xlsx"):
    """
    Scrapes articles from Kompas that match disaster-related keywords.
    Saves only new articles into an Excel file while keeping old ones, 
    and also updates the combined file.
    """
    url = "https://www.kompas.com/tag/bencana?page=1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    category_patterns = {
        "Banjir": r"\bbanjir\b|\bbanjir bandang\b",
        "Gempa Bumi": r"\bgempa\b|\bgempa bumi\b",
        "Tanah Longsor": r"\btanah longsor\b|\blongsor\b",
        "Gunung Meletus": r"\bgunung meletus\b|\berupsi\b",
        "Tsunami": r"\btsunami\b",
        "Puting Beliung": r"\bputing beliung\b|\bbadai\b",
        "Kekeringan": r"\bkekeringan\b|\bkemarau panjang\b",
        "Cuaca Ekstrem": r"\bcuaca ekstrem\b|\bhujan es\b|\bgelombang panas\b",
        "Gelombang Ekstrem": r"\bgelombang ekstrem\b|\bgemuruh laut\b",
        "Kebakaran Hutan": r"\bkebakaran hutan\b|\bapi di hutan\b|\bkarhutla\b"
    }
    keyword_pattern = re.compile("|".join(category_patterns.values()), re.IGNORECASE)
    
    print("\nScraping Kompas...")

    if os.path.exists(file_name):
        df_existing = pd.read_excel(file_name)
        existing_urls = set(df_existing["URL"].tolist())
    else:
        df_existing = pd.DataFrame(columns=["URL", "Date", "Title", "Content"])
        existing_urls = set()

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Kompas page: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('div', class_='article__list__title')
    if not articles:
        print("No articles found in Kompas.")
        return

    new_articles = []
    for article in articles:
        try:
            title = article.find('h3', class_='article__title').get_text(strip=True)
            link = article.find('a', class_='article__link')['href']
            date = article.find_next_sibling('div', class_='article__list__info') \
                          .find('div', class_='article__date') \
                          .get_text(strip=True)
            date = date.split(',')[0]
            if link in existing_urls:
                continue
            print(f"Processing: {title} (Kompas)")
            article_response = requests.get(link, headers=headers)
            if article_response.status_code != 200:
                print(f"Failed to access article: {link}")
                continue
            article_soup = BeautifulSoup(article_response.content, 'html.parser')
            paragraphs = []
            for p in article_soup.find_all('p'):
                if p.find('a', class_='inner-link-baca-juga'):
                    continue
                if p.find_parent('div', class_='footerCopyright'):
                    continue
                if not p.has_attr('class'):
                    text = ''.join([
                        elem.strip() if isinstance(elem, str)
                        else ' ' + elem.get_text(strip=True) + ' '
                        for elem in p.contents
                    ])
                    paragraphs.append(text)
            content = ' '.join(paragraphs)
            # Hapus teks sebelum dash pertama (jika bukan di tengah kata)
            for i, c in enumerate(content):
                if c in ('-', '–'):
                    left_is_alnum = (i > 0 and content[i-1].isalnum())
                    right_is_alnum = (i < len(content) - 1 and content[i+1].isalnum())
                    if left_is_alnum and right_is_alnum:
                        continue
                    content = content[i+1:].strip()
                    break
            if keyword_pattern.search(title) or keyword_pattern.search(content):
                new_articles.append({
                    'URL': link,
                    'Date': date,
                    'Title': title,
                    'Content': content
                })
        except Exception as e:
            print(f"Error processing article: {e}")

    if new_articles:
        print(f"Found {len(new_articles)} new articles in Kompas. Updating files...")
        df_new = pd.DataFrame(new_articles)
        df_kompas_combined = pd.concat([df_new, df_existing], ignore_index=True)
        df_kompas_combined.drop_duplicates(subset=["URL"], keep="first", inplace=True)
        df_kompas_combined.to_excel(file_name, index=False)
        print(f"File {file_name} updated with {len(new_articles)} new articles from Kompas.")
        update_gabungan(new_articles, gabungan_file)
    else:
        print("No new articles found from Kompas.")

def scrape_cnn(file_name="data_cnn.xlsx", gabungan_file="data_gabungan.xlsx"):
    """
    Scrapes CNN Indonesia articles for today's date that match disaster-related keywords.
    Saves only new articles into an Excel file while keeping old ones,
    and also updates the combined file.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    keywords = [
        r"\berupsi\b", r"\bgunung meletus\b", r"\btanah longsor\b", r"\blongsor\b",
        r"\bbanjir\b", r"\bbanjir bandang\b", r"\btsunami\b",
        r"\bgempa\b", r"\bgempa bumi\b", r"\bbadai\b", r"\bputing beliung\b", r"\bangin kencang\b",
        r"\bkekeringan\b", r"\bkemarau panjang\b", r"\bhujan es\b", r"\bgelombang panas\b",
        r"\bcuaca ekstrem\b", r"\bgelombang ekstrem\b", r"\bgemuruh laut\b",
        r"\bkebakaran hutan\b", r"\bkebakaran lahan\b", r"\bkarhutla\b", r"\bapi\b"
    ]
    keyword_pattern = re.compile("|".join(keywords), re.IGNORECASE)
    base_url = "https://www.cnnindonesia.com/peristiwa/indeks/18"
    print("\nScraping CNN Indonesia...")

    if os.path.exists(file_name):
        df_existing = pd.read_excel(file_name)
        existing_urls = set(df_existing["URL"].tolist())
    else:
        df_existing = pd.DataFrame(columns=["URL", "Date", "Title", "Content"])
        existing_urls = set()

    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching CNN page: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("article", class_="flex-grow")
    if not articles:
        print("No articles found in CNN.")
        return

    new_articles = []
    for article in articles:
        link_tag = article.find("a", href=True)
        link = link_tag['href'] if link_tag else None
        title_tag = article.find("h2", class_="text-cnn_black_light dark:text-white mb-2 inline leading-normal text-xl group-hover:text-cnn_red")
        title = title_tag.text.strip() if title_tag else "No title"
        if "FOTO" in title.upper():
            print(f"Skipping article (contains 'FOTO'): {title}")
            continue
        if not link or not link.startswith("http") or link in existing_urls:
            continue
        print(f"Processing: {title} (CNN)")
        try:
            article_response = requests.get(link, headers=headers)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(article_response.text, "html.parser")
            date_element = article_soup.find("div", class_="text-cnn_grey text-sm mb-4")
            date = date_element.text.strip() if date_element else "No date found"
            bulan_mapping = {
                "Jan": "Jan", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr", "Mei": "May",
                "Jun": "Jun", "Jul": "Jul", "Agu": "Aug", "Sep": "Sep", "Okt": "Oct",
                "Nov": "Nov", "Des": "Dec"
            }
            match = re.search(r'(\d{1,2}) (\w{3}) (\d{4})', date)
            if match:
                day, month, year = match.groups()
                month = bulan_mapping.get(month, month)
                date_obj = datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
                date = date_obj.strftime("%d/%m/%Y")
            content_element = article_soup.find("div", class_="detail-text text-cnn_black text-sm grow min-w-0")
            if content_element:
                for ad_div in content_element.find_all("div", class_="paradetail"):
                    ad_div.decompose()
                content_paragraphs = content_element.find_all("p")
                content = ""
                for p in content_paragraphs:
                    p_text = ""
                    for element in p.children:
                        if element.name == "span" or element.name == "a":
                            p_text += " " + element.get_text(strip=True) + " "
                        elif element.name is None:
                            p_text += element.strip() + " "
                    content += p_text.strip() + " "
                content = re.sub(r'\s+', ' ', content).strip()
            else:
                content = "No content found"
            if keyword_pattern.search(title) or keyword_pattern.search(content):
                new_articles.append({
                    "URL": link,
                    "Date": date,
                    "Title": title,
                    "Content": content
                })
            else:
                print(f"Skipping article (no matching keywords): {title}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching article {link}: {e}")

    if new_articles:
        print(f"Found {len(new_articles)} new articles in CNN. Updating files...")
        df_new = pd.DataFrame(new_articles)
        df_cnn_combined = pd.concat([df_new, df_existing], ignore_index=True)
        df_cnn_combined.drop_duplicates(subset=["URL"], keep="first", inplace=True)
        df_cnn_combined.to_excel(file_name, index=False)
        print(f"File {file_name} updated with {len(new_articles)} new articles from CNN.")
        update_gabungan(new_articles, gabungan_file)
    else:
        print("No new articles found from CNN.")

# 1. Fungsi prediksi jenis bencana
def predict_jenis(konten):
    jenis_cleaned = clean_text(konten)
    jenis_vectorized = vectorizer_jenis.transform([jenis_cleaned])
    jenis_prediction = model_jenis.predict(jenis_vectorized)
    # Konversi prediksi angka ke label string
    jenis_prediction = [label_mapping[pred] for pred in jenis_prediction]
    return jenis_prediction[0]  # Mengembalikan prediksi pertama

# 2. Fungsi NER dengan chunking dan cleaning
def ner_with_chunking_and_cleaning(text, max_length=512):
    """
    Melakukan NER pada teks panjang dengan pembagian chunk dan penanganan token hashtag (##).
    """
    def split_text_into_chunks(text, max_length):
        tokens = tokenizer.encode(text, truncation=False)
        chunks = []
        for i in range(0, len(tokens), max_length - 2):  # Ruang untuk [CLS] dan [SEP]
            chunk = tokens[i:i + max_length - 2]
            chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
            chunks.append(chunk)
        return chunks

    def clean_ner_results(results):
        cleaned_results = []
        for result in results:
            word = result['word']
            if word.startswith("##") and cleaned_results:
                cleaned_results[-1]['word'] += word[2:]
            else:
                cleaned_results.append(result)
        return cleaned_results

    ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    chunks = split_text_into_chunks(text, max_length)
    all_results = []
    for chunk in chunks:
        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
        ner_results = ner_pipe(decoded_chunk)
        cleaned_chunk_results = clean_ner_results(ner_results)
        all_results.extend(cleaned_chunk_results)
    return all_results

# Fungsi untuk mengekstrak entitas NER (lokasi, tanggal, waktu)
def extract_ner_entities(results):
    location = ""
    date = None
    time = None

    # Ekstraksi lokasi (GPE) secara berurutan maksimal 4 lokasi
    gpe_sequence = []
    gpe_found = False
    for result in results:
        entity_group = result.get("entity_group")
        word = result.get("word")
        if entity_group == "GPE":
            if not gpe_found:
                gpe_found = True
            if len(gpe_sequence) < 4:
                gpe_sequence.append(word)
            else:
                break
        elif gpe_found:
            break
    location = ", ".join([loc.title() for loc in gpe_sequence])
    
    # Ekstraksi tanggal (DAT), ambil yang pertama kali muncul
    for result in results:
        if result.get("entity_group") == "DAT" and date is None:
            word = result.get("word")
            try:
                match = re.search(r"(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})", word)
                if match:
                    cleaned_word = match.group(1).replace(" ", "")
                    date_obj = datetime.strptime(cleaned_word, "%d/%m/%Y")
                    date = date_obj.strftime("%d %B %Y")
                else:
                    match = re.search(r"(\d{1,2}\s*/\s*\d{1,2})", word)
                    if match:
                        cleaned_word = match.group(1).replace(" ", "")
                        current_year = datetime.now().year
                        cleaned_word += f"/{current_year}"
                        date_obj = datetime.strptime(cleaned_word, "%d/%m/%Y")
                        date = date_obj.strftime("%d %B")
            except ValueError:
                date = word

    # Ekstraksi waktu (TIM), ambil yang pertama kali muncul
    for result in results:
        if result.get("entity_group") == "TIM" and time is None:
            word = result.get("word")
            match = re.search(r"(\d{1,2})\.\s*(\d{2})\s*(wib|wita|wit)?", word, re.IGNORECASE)
            if match:
                hours = match.group(1)
                minutes = match.group(2)
                timezone = match.group(3).upper() if match.group(3) else ""
                time = f"{hours}.{minutes} {timezone}".strip()
            else:
                time = word
    if time is None:
        time = "Tidak ada dalam artikel"
    return location, date, time

# 3. Fungsi prediksi kalimat dampak bencana
def predict_impact(konten):
    kalimat_list = sentence_tokenize(konten)
    dampak_vectorized = vectorizer_dampak.transform(kalimat_list)
    dampak_pred = model_dampak.predict(dampak_vectorized)
    kalimat_berdampak = [kalimat for kalimat, label in zip(kalimat_list, dampak_pred) if label == 1]
    return " ".join(kalimat_berdampak)

def run_scrape_predict(interval=120):
    """
    Menjalankan proses scraping secara berurutan (Detik -> Kompas -> CNN) dalam satu siklus,
    kemudian melakukan prediksi pada artikel baru yang ada di data_gabungan.xlsx,
    dan menyimpan hasil prediksi ke data_prediksi.xlsx.
    
    Args:
        interval (int): Waktu tunggu (dalam detik) antara siklus.
    """
    try:
        while True:
            print("\n========== Starting New Scraping Cycle ==========\n")
            
            # Jalankan scraping dari masing-masing sumber berita
            scrape_detik()   # Fungsi ini meng-update file data_detik.xlsx dan data_gabungan.xlsx
            scrape_kompas()  # Fungsi ini meng-update file data_kompas.xlsx dan data_gabungan.xlsx
            scrape_cnn()     # Fungsi ini meng-update file data_cnn.xlsx dan data_gabungan.xlsx
            
            print("\n========== Scraping Completed. Starting Prediction Process ==========\n")
            
            # Muat ulang file data_gabungan.xlsx (mungkin telah terupdate dengan artikel baru)
            if os.path.exists("data_gabungan.xlsx"):
                df_gabungan = pd.read_excel("data_gabungan.xlsx")
            else:
                df_gabungan = pd.DataFrame(columns=["URL", "Date", "Title", "Content"])
            
            # Baca atau inisialisasi file data_prediksi.xlsx
            if os.path.exists("data_prediksi.xlsx"):
                df_prediksi = pd.read_excel("data_prediksi.xlsx")
            else:
                df_prediksi = pd.DataFrame(columns=["URL", "Type", "Location", "Date", "Time", "Impact"])
            
            # Iterasi untuk tiap baris di df_gabungan dan lakukan prediksi jika URL belum ada di df_prediksi
            for index, row in df_gabungan.iterrows():
                url = row.get("URL", None)
                if url is not None and url in df_prediksi['URL'].values:
                    continue  # Lewati jika URL sudah diprediksi
                
                konten = row['Content']
                
                # 1. Prediksi jenis bencana
                pred_type = predict_jenis(konten)
                
                # 2. Ekstraksi entitas NER: lokasi, tanggal, waktu
                ner_results = ner_with_chunking_and_cleaning(konten)
                location, date, time_extracted = extract_ner_entities(ner_results)
                
                # 3. Prediksi kalimat dampak bencana
                impact_text = predict_impact(konten)
                
                # Masukkan hasil prediksi ke df_prediksi
                new_row = {
                    "URL": url,
                    "Type": pred_type,
                    "Location": location,
                    "Date": date,
                    "Time": time_extracted,
                    "Impact": impact_text
                }
                df_prediksi = df_prediksi.append(new_row, ignore_index=True)
                
                # Simpan pembaruan ke file Excel
                df_prediksi.to_excel("data_prediksi.xlsx", index=False)
            
            print("Prediction process completed. Data prediksi telah diperbarui.")
            print(f"\n========== Cycle Completed. Waiting {interval // 60} minutes before next cycle ==========\n")
            t.sleep(interval)
    except KeyboardInterrupt:
        print("\nScraping and prediction stopped by user. Exiting safely.")

def run_scrape_predict_once():
    # Jalankan scraping dari masing-masing sumber
    scrape_detik()
    scrape_kompas()
    scrape_cnn()

    # Baca data gabungan
    if os.path.exists("data_gabungan.xlsx"):
        df_gabungan = pd.read_excel("data_gabungan.xlsx")
    else:
        df_gabungan = pd.DataFrame(columns=["URL", "Date", "Title", "Content"])
    
    # Baca atau inisialisasi data_prediksi
    if os.path.exists("data_prediksi.xlsx"):
        df_prediksi = pd.read_excel("data_prediksi.xlsx")
    else:
        df_prediksi = pd.DataFrame(columns=["URL", "Type", "Location", "Date", "Time", "Impact"])
    
    # Proses prediksi untuk artikel baru
    for index, row in df_gabungan.iterrows():
        url = row.get("URL", None)
        if url is not None and url in df_prediksi['URL'].values:
            continue
        konten = row['Content']
        pred_type = predict_jenis(konten)
        ner_results = ner_with_chunking_and_cleaning(konten)
        location, date, time_extracted = extract_ner_entities(ner_results)
        impact_text = predict_impact(konten)
        new_row = {
            "URL": url,
            "Type": pred_type,
            "Location": location,
            "Date": date,
            "Time": time_extracted,
            "Impact": impact_text
        }
        df_prediksi = df_prediksi.append(new_row, ignore_index=True)
    
    df_prediksi.to_excel("data_prediksi.xlsx", index=False)
    print("Satu siklus scraping dan prediksi selesai.")

# Fungsi geocode: mengubah lokasi menjadi koordinat
def get_coordinates(location):
    geolocator = Nominatim(user_agent="disaster_locator")
    try:
        t.sleep(1)  # Jeda 1 detik untuk menghindari rate limit Nominatim
        location_data = geolocator.geocode(location)
        if location_data:
            print(f"Geocoded {location}: {location_data.latitude}, {location_data.longitude}")
            return location_data.latitude, location_data.longitude
        else:
            print(f"Geocoding failed for: {location}")
            return None, None
    except Exception as e:
        print(f"Error geocoding {location}: {e}")
        return None, None

def create_map(df_prediksi):
    print("Mencari koordinat lokasi...")
    df_prediksi[['Latitude', 'Longitude']] = df_prediksi['Location'].apply(lambda x: pd.Series(get_coordinates(x)))

    print("Membuat peta...")
    m = folium.Map(location=[-2.5489, 118.0149], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df_prediksi.iterrows():
        if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
            # Ambil satu kalimat pertama dari Impact
            impact_text = str(row['Impact'])  # pastikan string
            sentences = re.split(r'[.!?]', impact_text)  # pisahkan berdasarkan . ! ?
            first_sentence = sentences[0].strip() if sentences else ""

            # Tambahkan tanda titik di akhir jika belum ada
            if first_sentence and not first_sentence.endswith('.'):
                first_sentence += '.'

            popup_text = f"""
            <b>Type:</b> {row['Type']}<br>
            <b>Location:</b> {row['Location']}<br>
            <b>Date:</b> {row['Date']}<br>
            <b>Time:</b> {row['Time']}<br>
            <b>Impact:</b> {first_sentence}
            <br><a href='{row['URL']}' target='_blank'>Baca Selengkapnya</a>
            """

            icon_color = "red" if "Banjir" in row['Type'] else "blue"
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=icon_color)
            ).add_to(marker_cluster)

    map_path = "static/map_bencana.html"
    m.save(map_path)
    print(f"Peta berhasil disimpan: {map_path}")
    return map_path


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/team')
def team():
    return render_template("team.html")

@app.route('/update_data', methods=['GET'])
def update_data():
    # Panggil fungsi 1 siklus update
    run_scrape_predict_once()
    # Kembalikan respon JSON saat selesai
    return jsonify({"message": "Scraping and extraction finished."})

@app.route('/updates')
def updates():
    if os.path.exists("data_prediksi.xlsx"):
        df_prediksi = pd.read_excel("data_prediksi.xlsx")

        # Ganti NaN di kolom Impact dengan 'Tidak ada'
        df_prediksi['Impact'] = df_prediksi['Impact'].fillna('Tidak ada')

        # Ganti beberapa literal string tak valid di kolom Date dengan NaT
        df_prediksi['Date'] = df_prediksi['Date'].replace(
            ['nan', 'Unknown Date', 'No date found'],
            pd.NaT
        )

        # Fungsi untuk menambahkan tahun berjalan jika belum ada
        def add_current_year_if_missing(date_str):
            """
            Jika date_str cocok format "dd Month" (tanpa tahun),
            tambahkan tahun berjalan di belakang.
            Contoh: "23 February" -> "23 February 2025" (jika tahun sekarang 2025)
            """
            if isinstance(date_str, str):
                pattern = r'^\d{1,2}\s+[A-Za-z]+(\s+\d{4})?$'
                if re.match(pattern, date_str):
                    if not re.search(r'\d{4}', date_str):
                        current_year = datetime.now().year
                        date_str = f"{date_str} {current_year}"
            return date_str

        # Terapkan fungsi di atas ke setiap baris di kolom Date
        df_prediksi['Date'] = df_prediksi['Date'].apply(
            lambda x: add_current_year_if_missing(x) if pd.notnull(x) else x
        )

        # Konversi kolom Date ke datetime (format "dd Month yyyy")
        df_prediksi['Date'] = pd.to_datetime(
            df_prediksi['Date'],
            format='%d %B %Y',
            errors='coerce'
        )

        # ------------------------------------------
        # 1) Ambil parameter filter dari query string
        # ------------------------------------------
        disaster_type = request.args.get('disaster_type', '')
        start_date_str = request.args.get('start_date', '')
        end_date_str = request.args.get('end_date', '')

        # 2) Filter Jenis Bencana
        if disaster_type:
            df_prediksi = df_prediksi[df_prediksi['Type'] == disaster_type]

        # 3) Filter Tanggal (start_date, end_date) format "yyyy-mm-dd"
        if start_date_str:
            try:
                start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
                df_prediksi = df_prediksi[df_prediksi['Date'] >= start_dt]
            except ValueError:
                pass  # jika parsing gagal, abaikan filter

        if end_date_str:
            try:
                end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
                df_prediksi = df_prediksi[df_prediksi['Date'] <= end_dt]
            except ValueError:
                pass

        # 4) Urutkan berdasarkan Date (terbaru -> terlama), NaT di akhir
        df_prediksi.sort_values(by='Date', ascending=False, na_position='last', inplace=True)

        # 5) Ubah kembali kolom Date ke string dd/mm/yyyy. NaT -> 'Tidak ada'
        df_prediksi['Date'] = df_prediksi['Date'].apply(
            lambda x: x.strftime('%d/%m/%Y') if pd.notnull(x) else 'Tidak ada'
        )

        # 6) Ubah dataframe ke list of dict
        data = df_prediksi.to_dict(orient='records')
    else:
        data = []

    # ---------------------------------------
    # 7) Pagination (10 data per halaman)
    # ---------------------------------------
    page = request.args.get('page', 1, type=int)
    per_page = 10
    total = len(data)
    total_pages = math.ceil(total / per_page) if total > 0 else 1

    start = (page - 1) * per_page
    end = start + per_page
    page_data = data[start:end]

    # Kirim data subset + info pagination + filter ke template
    return render_template(
        "updates.html",
        predictions=page_data,
        page=page,
        total_pages=total_pages,
        total=total
    )

@app.route('/visual')
def visual():
    # Jika file data_prediksi.xlsx tidak ada, kirim chart kosong
    if not os.path.exists("data_prediksi.xlsx"):
        return render_template(
            "visual.html",
            chart_labels=[],
            chart_values=[],
            chart_colors=[],
            chart_border_colors=[]
        )

    # Baca file
    df_prediksi = pd.read_excel("data_prediksi.xlsx")

    # Bersihkan data
    df_prediksi['Impact'] = df_prediksi['Impact'].fillna('Tidak ada')
    df_prediksi['Date'] = df_prediksi['Date'].replace(['nan', 'Unknown Date', 'No date found'], pd.NaT)

    def add_current_year_if_missing(date_str):
        """
        Jika date_str hanya "dd Month", tambahkan tahun berjalan.
        Contoh: "23 February" -> "23 February 2025"
        """
        if isinstance(date_str, str):
            pattern = r'^\d{1,2}\s+[A-Za-z]+(\s+\d{4})?$'
            if re.match(pattern, date_str):
                if not re.search(r'\d{4}', date_str):
                    current_year = datetime.now().year
                    date_str = f"{date_str} {current_year}"
        return date_str

    # Terapkan fungsi di atas
    df_prediksi['Date'] = df_prediksi['Date'].apply(
        lambda x: add_current_year_if_missing(x) if pd.notnull(x) else x
    )
    # Konversi ke datetime
    df_prediksi['Date'] = pd.to_datetime(df_prediksi['Date'], format='%d %B %Y', errors='coerce')

    # Ambil parameter filter
    # Gunakan getlist agar bisa menampung beberapa bencana (multi-select)
    disaster_types = request.args.getlist('disaster_type')  
    start_date_str = request.args.get('start_date', '')
    end_date_str = request.args.get('end_date', '')

    # Filter jenis bencana (jika user memilih bencana tertentu)
    if disaster_types:
        df_prediksi = df_prediksi[df_prediksi['Type'].isin(disaster_types)]

    # Filter tanggal (format input date: yyyy-mm-dd)
    if start_date_str:
        try:
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
            df_prediksi = df_prediksi[df_prediksi['Date'] >= start_dt]
        except ValueError:
            pass

    if end_date_str:
        try:
            end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            df_prediksi = df_prediksi[df_prediksi['Date'] <= end_dt]
        except ValueError:
            pass

    # Agregasi: jumlah artikel per jenis bencana
    df_agg = df_prediksi.groupby('Type').size().reset_index(name='count')
    df_agg.sort_values(by='count', ascending=False, inplace=True)

    # Buat list label dan values untuk chart
    chart_labels = df_agg['Type'].tolist()
    chart_values = df_agg['count'].tolist()

    # Mapping warna untuk setiap jenis bencana
    color_map = {
        "Banjir": "rgba(54, 162, 235, 0.6)",
        "Bencana Hidrometerologi Ekstrem": "rgba(255, 99, 132, 0.6)",
        "Gempa Bumi": "rgba(255, 206, 86, 0.6)",
        "Gunung Meletus": "rgba(75, 192, 192, 0.6)",
        "Puting Beliung": "rgba(153, 102, 255, 0.6)",
        "Tanah Longsor": "rgba(255, 159, 64, 0.6)",
        "Tsunami": "rgba(199, 199, 199, 0.6)"
    }
    border_color_map = {
        "Banjir": "rgba(54, 162, 235, 1)",
        "Bencana Hidrometerologi Ekstrem": "rgba(255, 99, 132, 1)",
        "Gempa Bumi": "rgba(255, 206, 86, 1)",
        "Gunung Meletus": "rgba(75, 192, 192, 1)",
        "Puting Beliung": "rgba(153, 102, 255, 1)",
        "Tanah Longsor": "rgba(255, 159, 64, 1)",
        "Tsunami": "rgba(199, 199, 199, 1)"
    }

    # Buat list warna sesuai label
    chart_colors = [color_map.get(lbl, "rgba(0,0,0,0.6)") for lbl in chart_labels]
    chart_border_colors = [border_color_map.get(lbl, "rgba(0,0,0,1)") for lbl in chart_labels]

    # Kirim ke template
    return render_template(
        "visual.html",
        chart_labels=chart_labels,
        chart_values=chart_values,
        chart_colors=chart_colors,
        chart_border_colors=chart_border_colors
    )

@app.route('/map')
def map_view():
    if not os.path.exists("data_prediksi.xlsx"):
        return "Data tidak tersedia."
    
    df_prediksi = pd.read_excel("data_prediksi.xlsx")
    # Lakukan filtering atau pembersihan data jika diperlukan
    # Misalnya, pastikan kolom 'Location' dan 'Title' ada.
    map_path = create_map(df_prediksi)
    return render_template("map.html", map_path=map_path)


if __name__ == '__main__':
    app.run(debug=True)