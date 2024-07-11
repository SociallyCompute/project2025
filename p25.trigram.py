import PyPDF2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import string
import re

# Ensure you have the required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def read_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def clean_text(text):
    # Tokenize text
    words = nltk.word_tokenize(text.lower())
    # Remove punctuation, stopwords, months, numbers, and "https"
    stop_words = set(stopwords.words('english'))
    months = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
    words = [
        word for word in words 
        if word.isalnum() and 
        word not in stop_words and 
        word not in months and 
        not word.isdigit() and 
        "https" not in word
    ]
    return words

def get_trigrams(words):
    trigrams = list(ngrams(words, 3))
    trigram_freq = Counter(trigrams)
    return trigram_freq

def create_word_cloud_from_trigrams(trigram_freq, output_image_path):
    # Convert trigram tuples to strings
    trigram_dict = {' '.join(trigram): freq for trigram, freq in trigram_freq.items()}
    wordcloud = WordCloud(width=2400, height=1200, background_color='white').generate_from_frequencies(trigram_dict)
    plt.figure(figsize=(30, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_image_path)
    plt.show()

if __name__ == "__main__":
    pdf_path = './2025_MandateForLeadership_FULL.pdf'
    output_image_path = 'trigram_word_cloud.png'
    
    text = read_pdf(pdf_path)
    words = clean_text(text)
    trigram_freq = get_trigrams(words)
    create_word_cloud_from_trigrams(trigram_freq, output_image_path)
