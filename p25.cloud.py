import PyPDF2
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def read_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def create_word_cloud(text, output_image_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_image_path)
    plt.show()

if __name__ == "__main__":
    pdf_path = './2025_MandateForLeadership_FULL.pdf'
    output_image_path = 'word_cloud.png'
    
    text = read_pdf(pdf_path)
    create_word_cloud(text, output_image_path)
