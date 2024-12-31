# UseCaseGenie - GenAI Use Case Discovery

**Use Case Genie** is a powerful GenAI application designed to help you discover and explore relevant use cases for your company or industry. This application leverages advanced language models and web scraping to provide insightful reports on the potential of Generative AI in various contexts.

## Features

*   **Intelligent Query Analysis**: Automatically identifies if your query refers to a specific company or an industry.
*   **Web Research**: Scrapes web content to gather relevant information based on your query.
*   **Use Case Generation**: Employs expert analyst personas to generate innovative use cases for GenAI in your chosen area.
*   **Structured Reporting**: Delivers a well-structured markdown report of the generated use cases.
*   **Dataset Feasibility Analysis:** (Commented out, but provides a framework for potential future functionality) Explores dataset availability by generating relevant keywords and searching for datasets. 
*   **Downloadable Report**: Generates downloadable markdown report with all the analysis.

## Getting Started

### Prerequisites

Before you begin, make sure you have the following installed:

*   **Python 3.8+**
*   **pip** (Python package manager)

You'll also need API keys for the following services (you will be prompted when you run the application if the .env file is not setup):

*   **Groq API Key** - Required to use Groq LLM
*   **Tavily API Key** - Required to use Tavily for web search
*   **LangChain API Key** - Used for Langchain
*   **Jina API Key** - Required for Jina Embeddings

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**
     Create a `.env` file in the root directory of your project and add your API keys.

     ```
    GROQ_API_KEY=your_groq_api_key
    TAVILY_API_KEY=your_tavily_api_key
    LANGCHAIN_API_KEY=your_langchain_api_key
    JINA_API_KEY=your_jina_api_key
    ```

### Usage

1.  **Run the main script:**

    ```bash
    python main.py
    ```

2.  **Access the Gradio Interface:**
   Open your browser and navigate to the address shown in the terminal (typically http://127.0.0.1:7860 or similar).

3. **Input your Query:**
    Enter your query about a specific company or industry in the text box.
4. **Submit the Query:**
    Click the 'Submit' button to start the analysis and use case generation.
5.  **Download the Report:**
    Download the generated markdown report to view the results.

## File Structure

*   `agent.py`: Contains the logic for the research agent, which analyzes user queries and scrapes the web.
*   `setup.py`: Handles environment variables and initializes the LLM and search tool.
*   `usecase_agent.py`: Defines the agent responsible for generating use cases.
*   `feasibility_agent.py`: (commented out) Handles keyword generation for datasets.
*   `vectorstore.py`: Manages the document loading and processing, including cleaning and chunking
*   `main.py`: Sets up the Gradio interface and orchestrates the overall process.
*   `requirements.txt`: Lists the Python dependencies.

## Code Details

### Agent Workflow
1. **Research Agent (`agent.py`)**:
    *   Receives a user query.
    *   Uses an LLM to determine if the query is about a company, an industry, or both.
    *   Constructs search queries for web scraping based on the extracted company and/or industry names.
    *   Gathers relevant links and content.

2.  **Vector Store (`vectorstore.py`)**:
    *   Categorizes URLs into HTML and PDF files based on file type.
    *   Extracts and cleans text from web pages.
    *   Splits extracted text into chunks for context.
    *   Writes document chunks into a vector database for retrieval.

3.  **Usecase Agent (`usecase_agent.py`)**:
    *   Creates multiple AI analyst personas with specific roles.
    *   Conducts interviews with these personas, asking them about the implications of GenAI based on the scraped context.
    *   Based on the interviews, generates structured use case memos.
    *   The use case memos are consolidated into a final report, including a compelling introduction and conclusion.

4.  **Feasibility Agent (`feasibility_agent.py`)**: *(Note: This is commented out, meaning this functionality is not fully integrated)*
    *   Generates keywords from each use case.
    *   Searches for datasets on platforms such as Kaggle, Github, and Huggingface.
    *   Combines results and returns an excel sheet with use case and a list of links for dataset.

5.  **Main (`main.py`)**:
    *   Sets up the Gradio interface to take user input.
    *   Triggers the research agent, vectorstore loading, and use-case generation logic.
    *   Displays the final report through the Gradio interface.
    *   Manages the file downloads for the generated report.

## Potential improvements

*   **Feasibility Analysis**: Complete integration of the feasibility agent to check dataset availability.
*   **GUI Enhancement**: Add a better and user friendly GUI.
*   **Documentation**: More detailed documentation on each module and functionality.
*   **Error Handling**: Add more detailed error handling and logging.


