import PyPDF2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import string

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
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

def get_bigrams(words):
    bigrams = list(ngrams(words, 2))
    bigram_freq = Counter(bigrams)
    return bigram_freq

def create_word_cloud_from_bigrams(bigram_freq, output_image_path):
    # Convert bigram tuples to strings
    bigram_dict = {' '.join(bigram): freq for bigram, freq in bigram_freq.items()}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bigram_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_image_path)
    plt.show()

if __name__ == "__main__":
    pdf_path = './2025_MandateForLeadership_FULL.pdf'
    output_image_path = 'bigram_word_cloud.png'
    
    text = read_pdf(pdf_path)
    words = clean_text(text)
    bigram_freq = get_bigrams(words)
    create_word_cloud_from_bigrams(bigram_freq, output_image_path)
