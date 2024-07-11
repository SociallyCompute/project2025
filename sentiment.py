import PyPDF2
from transformers import pipeline, AutoTokenizer
import matplotlib.pyplot as plt
import nltk

# Ensure you have the required NLTK data
nltk.download('punkt')

def read_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def split_into_sections(text, num_sections):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    # Split sentences into roughly equal sections
    section_size = len(sentences) // num_sections
    sections = [sentences[i * section_size:(i + 1) * section_size] for i in range(num_sections)]
    sections = [' '.join(section) for section in sections]
    return sections

def split_into_chunks(text, max_length=512):
    # Split text into chunks ensuring each chunk is within the model's max token limit
    tokens = text.split()
    chunks = []
    chunk = []
    chunk_length = 0

    for token in tokens:
        token_length = len(token)
        if chunk_length + token_length + 1 > max_length:  # +1 for space or special token
            chunks.append(' '.join(chunk))
            chunk = [token]
            chunk_length = token_length
        else:
            chunk.append(token)
            chunk_length += token_length + 1

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

def analyze_sentiment(sections):
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_scores = []
    for section in sections:
        chunks = split_into_chunks(section)
        section_scores = []
        for chunk in chunks:
            result = sentiment_analyzer(chunk)
            # Average the scores for the chunk
            avg_score = sum([r['score'] if r['label'] == 'POSITIVE' else -r['score'] for r in result]) / len(result)
            section_scores.append(avg_score)
        # Average the scores for the section
        section_avg_score = sum(section_scores) / len(section_scores)
        sentiment_scores.append(section_avg_score)
    return sentiment_scores

def visualize_sentiment(sentiment_scores, output_image_path):
    sections = range(1, len(sentiment_scores) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(sections, sentiment_scores, marker='o')
    plt.title('Sentiment Analysis of Project 2025\'s Giant Blob of Sinister by Section')
    plt.xlabel('Section')
    plt.ylabel('Sentiment Score')
    plt.grid(True)

    # Add a caption below the plot
    caption = "Figure 1: Sentiment analysis scores for each section of the document. Notice that the document begins with positive sentiment. This fishes the reader in, makes them feel ok. People like happy. Then they drop the hammer of fear and hate."
    plt.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=12)

    plt.savefig(output_image_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    pdf_path = './2025_MandateForLeadership_FULL.pdf'
    output_image_path = 'sentiment_analysis_by_section.png'
    num_sections = 10  # Adjust the number of sections as needed
    
    text = read_pdf(pdf_path)
    sections = split_into_sections(text, num_sections)
    sentiment_scores = analyze_sentiment(sections)
    visualize_sentiment(sentiment_scores, output_image_path)
