from setup import *
import tempfile
import requests

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import urlparse
from langchain.docstore.document import Document



def extract_urls(agentstate_result):
  urls = []
  content=[]
  for item in agentstate_result['link_list']:
    urls.append(item['url'])
    content.append(item['content'])
  
  return urls, content



# Function to classify URL based on file extension
def classify_url_by_extension(url):
    """
    Classifies a URL based on its file extension.
    Focuses only on pdf and html, classifying others as unknown.
    """

    if not isinstance(url, str):
        raise ValueError(f"Expected a string, but got {type(url)}")

    # Extract the file extension from the URL
    try:
        file_extension = urlparse(url).path.split('.')[-1].lower()
        if file_extension == 'pdf':
            return 'pdf'
        elif file_extension in ['html', 'htm']:
            return 'html'
        else:
            return 'unknown'
    except Exception as e:
        print(f"Error while parsing URL: {url} - {e}")
        return 'unknown'


# Function to classify based on HTTP Content-Type header (optional, for extra accuracy)
def classify_url_by_header(url):
    """
    Classifies a URL based on the HTTP Content-Type header.
    Focuses only on pdf and html, classifying others as unknown.
    """
    try:
        response = requests.head(url, timeout=5)  # Use HEAD request to fetch headers
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'pdf' in content_type:
            return 'pdf'
        elif 'html' in content_type:
            return 'html'
        else:
            return 'unknown'
    except requests.RequestException as e:
        print(f"Error while making HEAD request: {url} - {e}")
        return 'unknown'


# Function to classify a list of URLs
def urls_classify_list(urls):
    """
    Classifies a list of URLs into pdf, html, and unknown.
    Returns two separate lists: one for pdf URLs and one for html URLs.
    """
    if not isinstance(urls, list):
        raise ValueError("Expected a list of URLs")

    pdf_urls = []
    html_urls = []

    # Classify each URL
    for url in urls:
        file_type = classify_url_by_extension(url)  # First, try classifying by extension
        if file_type == 'unknown':
            # If extension-based classification failed, fall back to HTTP header classification
            file_type = classify_url_by_header(url)

        if file_type == 'pdf':
            pdf_urls.append(url)
        elif file_type == 'html':
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
        
    return pdf_urls, html_urls



def clean_and_extract_html_data(html_urls, chunk_size=100, chunk_overlap=25):
    """
    Loads HTML content from URLs, cleans the data, and splits it into smaller chunks.

    Args:
        html_urls (list): List of HTML URLs to process.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of document chunks.
    """

    def clean_content(content):
        """
        Cleans the content by removing unwanted patterns and short lines.
        """
        cleaned_content = content.strip()  # Remove leading/trailing whitespace
        lines = cleaned_content.split('\n')  # Split by newlines
        meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 3]  # Keep meaningful lines
        return '\n'.join(meaningful_lines)

    def split_document(doc_content, chunk_size, chunk_overlap):
        """
        Splits a document into smaller chunks with overlap.
        """
        chunks = []
        start = 0
        while start < len(doc_content):
            end = start + chunk_size
            chunk = doc_content[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap if end < len(doc_content) else len(doc_content)
        return chunks

    # Step 1: Load documents from URLs
    docs = []
    for url in html_urls:
        try:
            loader = WebBaseLoader(url)
            data = loader.load()
            docs.extend(data)
        except Exception as e:
            print(f"Error loading URL {url}: {e}")

    # Step 2: Clean the content to remove unwanted data
    cleaned_docs = []
    for doc in docs:
        cleaned_content = clean_content(doc.page_content)
        if cleaned_content:  # Exclude empty documents
            doc.page_content = cleaned_content
            cleaned_docs.append(doc)

    # Step 3: Split the cleaned documents into chunks
    doc_splits = []
    for doc in cleaned_docs:
        chunks = split_document(doc.page_content, chunk_size, chunk_overlap)
        for chunk in chunks:
            doc_splits.append(Document(page_content=chunk, metadata=doc.metadata))

    return doc_splits






# def extract_pdf_from_url(url):
#     """
#     Extract text from a PDF available at a URL.
    
#     Args:
#         url (str): The URL of the PDF file.
        
#     Returns:
#         str: Extracted text from the PDF.
#     """
#     # Step 1: Download the PDF from the URL
#     response = requests.get(url)
#     if response.status_code == 200:
#         pdf_content = response.content
#     else:
#         raise ValueError(f"Failed to fetch the PDF. HTTP Status Code: {response.status_code}")
    
#     # Step 2: Save PDF content to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#         temp_pdf.write(pdf_content)
#         temp_pdf_path = temp_pdf.name  # Get the file path
    
#     # Step 3: Load the PDF using PyPDFLoader
#     loader = PyPDFLoader(temp_pdf_path)
#     documents = loader.load()
    
#     # Step 4: Extract text from all pages
#     extracted_text = "\n".join(doc.page_content for doc in documents)
    
#     return extracted_text


# def clean_and_split_pdf_text(pdf_text, chunk_size=100, chunk_overlap=25):
#     """
#     Cleans and splits the extracted PDF text into smaller chunks.
    
#     Args:
#         pdf_text (str): Extracted text from a PDF.
#         chunk_size (int): Maximum size of each chunk.
#         chunk_overlap (int): Overlap between chunks.
        
#     Returns:
#         list: List of document chunks.
#     """
#     def clean_content(content):
#         """
#         Cleans the text by removing unwanted patterns and short lines.
#         """
#         content = content.strip()  # Remove leading/trailing whitespace
#         lines = content.split('\n')  # Split into lines
#         meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 3]  # Exclude short lines
#         return '\n'.join(meaningful_lines)

#     def split_text(content, chunk_size, chunk_overlap):
#         """
#         Splits cleaned text into smaller chunks with overlap.
#         """
#         chunks = []
#         start = 0
#         while start < len(content):
#             end = start + chunk_size
#             chunks.append(content[start:end])
#             start = end - chunk_overlap if end < len(content) else len(content)
#         return chunks

#     # Step 1: Clean the text
#     cleaned_text = clean_content(pdf_text)

#     # Step 2: Split the cleaned text
#     return split_text(cleaned_text, chunk_size, chunk_overlap)


# def pdf_extraction(pdf_urls, chunk_size=100, chunk_overlap=25):
#     """
#     Extracts and processes text from a list of PDF URLs.
    
#     Args:
#         pdf_urls (list): List of PDF URLs.
#         chunk_size (int): Maximum size of each chunk.
#         chunk_overlap (int): Overlap between chunks.
        
#     Returns:
#         list: List of Document objects containing split text.
#     """
#     all_chunks = []
    
#     for pdf_url in pdf_urls:
#         try:
#             # Extract text from the PDF
#             extracted_text = extract_pdf_from_url(pdf_url)
            
#             # Clean and split the text
#             chunks = clean_and_split_pdf_text(extracted_text, chunk_size, chunk_overlap)
            
#             # Convert chunks into Document objects
#             for chunk in chunks:
#                 all_chunks.append(Document(page_content=chunk, metadata={"source": pdf_url}))
#         except Exception as e:
#             print(f"Error processing PDF URL {pdf_url}: {e}")
    
#     return all_chunks
