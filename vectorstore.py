from setup import *
import tempfile
import requests

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import urlparse


def extract_urls(agentstate_result):
  urls = []
  content=[]
  for item in agentstate_result['link_list']:
    print(item)
    urls.append(item['url'])
    content.append(item['content'])
  
  return urls, content



# Function to classify URL based on file extension
def classify_url_by_extension(url):
    # Define possible file types and their extensions
    file_types = {
        'pdf': ['.pdf'],
        'html': ['.html', '.htm'],
        'text': ['.txt'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'pptx': ['.pptx', '.ppt'],
        'docx': ['.docx', '.doc'],
        'xlsx': ['.xlsx', '.xls'],
    }

    # Extract the file extension from the URL
    file_extension = urlparse(url).path.split('.')[-1].lower()

    for category, extensions in file_types.items():
        if f'.{file_extension}' in extensions:
            return category
    return 'unknown'


# Function to classify based on HTTP Content-Type header (optional, for extra accuracy)
def classify_url_by_header(url):
    try:
        response = requests.head(url, timeout=5)  # Use HEAD request to get only headers
        content_type = response.headers.get('Content-Type', '').lower()

        if 'pdf' in content_type:
            return 'pdf'
        elif 'html' in content_type:
            return 'html'
        elif 'image' in content_type:
            return 'image'
        elif 'pptx' in content_type or 'presentation' in content_type:
            return 'pptx'
        elif 'word' in content_type or 'msword' in content_type:
            return 'docx'
        elif 'excel' in content_type:
            return 'xlsx'
        else:
            return 'unknown'
    except requests.RequestException:
        return 'unknown'


def urls_classify_list(urls: list):
    pdf_urls=[]
    html_urls=[]
    # Classify the URLs
    for url in urls:
        file_type = classify_url_by_extension(url)  # First, try classifying by extension
        if file_type == 'unknown':
            # If extension-based classification failed, fall back to HTTP header classification
            file_type = classify_url_by_header(url)
        
        if file_type == 'pdf':
            pdf_urls.append(url)
        
        if file_type == 'html' or file_type == 'unknown':
            html_urls.append(url)
        
    return pdf_urls, html_urls


def urls_classify_list(urls: list):
    pdf_urls=[]
    html_urls=[]
    # Classify the URLs
    for url in urls:
        file_type = classify_url_by_extension(url)  # First, try classifying by extension
        if file_type == 'unknown':
            # If extension-based classification failed, fall back to HTTP header classification
            file_type = classify_url_by_header(url)
        
        if file_type == 'pdf':
            pdf_urls.append(url)
        
        if file_type == 'html' or file_type == 'unknown':
            html_urls.append(url)
        
    print('---PDF--', pdf_urls, html_urls)
    return pdf_urls, html_urls



def clean_and_extract_html_data(html_urls):
    # Step 1: Load documents from URLs
    docs = []
    for url in html_urls:
        loader = WebBaseLoader(url)
        data = loader.load()
        docs.extend(data)
    
    # Step 2: Clean the content to remove unwanted data
    cleaned_docs = []
    for doc in docs:
        cleaned_content = doc.page_content.strip()  # Remove leading/trailing whitespace
        
        # Remove specific patterns or redundant lines
        lines = cleaned_content.split('\n')  # Split by newlines
        meaningful_lines = [line for line in lines if len(line.strip()) > 3]  # Exclude short or empty lines
        
        # Reconstruct cleaned content
        cleaned_content = '\n'.join(meaningful_lines)
        if cleaned_content:  # Exclude empty documents
            doc.page_content = cleaned_content
            cleaned_docs.append(doc)
    
    # Step 3: Split the cleaned documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(cleaned_docs)
    
    return doc_splits




def extract_pdf_from_url(url):
    """
    Extract text from a PDF available at a URL using LangChain's PyPDFLoader.
    
    Args:
        url (str): The URL of the PDF file.
        
    Returns:
        str: Extracted text from the PDF.
    """
    # Step 1: Download the PDF from the URL
    response = requests.get(url)
    if response.status_code == 200:
        pdf_content = response.content
    else:
        raise ValueError(f"Failed to fetch the PDF. HTTP Status Code: {response.status_code}")
    
    # Step 2: Save PDF content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_content)
        temp_pdf_path = temp_pdf.name  # Get the file path
    
    # Step 3: Load the PDF using PyPDFLoader
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()
    
    # Step 4: Extract text from all pages
    extracted_text = "\n".join(doc.page_content for doc in documents)
    
    return extracted_text


def pdf_extraction(pdf_urls):
  extracted_text = [extract_pdf_from_url(pdf_url) for pdf_url in pdf_urls]
  docs_list = [item for sublist in extracted_text for item in sublist]
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  doc_splits = text_splitter.create_documents(docs_list)
  return doc_splits