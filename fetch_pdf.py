import os
import PyPDF2
import requests
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')

file_path = './Green Energy Papers Database.xlsx'  # Update with the actual file path
data = pd.read_excel(file_path)

def preprocess_text(text):
    text = text.lower()
    
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    words = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

def download_pdf(doi_url, output_path):
    response = requests.get(doi_url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfFileReader(f)
        num_pages = reader.numPages
        text = ""
        for page in range(num_pages):
            text += reader.getPage(page).extractText()
    return text

pdf_dir = './'  # Update with the actual directory path
os.makedirs(pdf_dir, exist_ok=True)

doi_urls = data['DOI'].tolist()
abstracts = []

for idx, doi_url in enumerate(doi_urls):
    try:
        pdf_path = os.path.join(pdf_dir, f'paper_{idx+1}.pdf')
        download_pdf(doi_url, pdf_path)
        raw_text = extract_text_from_pdf(pdf_path)
        preprocessed_text = preprocess_text(raw_text)
        abstracts.append(preprocessed_text)
    except Exception as e:
        print(f"Error processing {doi_url}: {e}")
        abstracts.append("")