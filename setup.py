import os
import getpass
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults

load_dotenv()

def _set_env(var: str):
  if not os.environ.get(var):
    os.environ[var] = getpass.getpass(f"Enter {var}: ")


# Setting up environmentl_variables
_set_env("GROQ_API_KEY")
_set_env("TAVILY_API_KEY")
_set_env("LANGCHAIN_API_KEY")
_set_env("JINA_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"



# llm defined
llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")

# search tool
tavily_search = TavilySearchResults(
    max_results=2,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)
