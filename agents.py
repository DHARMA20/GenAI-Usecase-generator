from setup import *
import re
import requests
from typing import Annotated, Sequence, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# Research agent
class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]
  queries : List[str]
  link_list : Optional[List]
  industry : Optional[str]
  company: Optional[str]



# Node
def assistant(state: AgentState):
  assistant_sys_msg = SystemMessage(content='''You are a highly intelligent and helpful assistant. Your primary task is to analyze user queries and determine whether the query:

    Refers to an industry (general context)
    Refers to a specific company (e.g., mentions a company's name explicitly).

    For every query:
    Check for company names, brands, or proper nouns that indicate a specific entity.
    While analyzing the company industry be specific as possible.
    Return the company and industry name in the query
    if you can't find a industry name, return an empty string.
                            
    Example 1:
    Query: "GenAI in MRF Tyres"
    Company: "MRF Tyres"
    Industry: "Tires and rubber products"

    Example 2:  
    Query: "GenAI in the healthcare industry"
    Company: ""
    Industry: "Healthcare"
    ''')
  return {'messages': [llm.invoke([assistant_sys_msg] + state["messages"])]}



def company_and_industry_query(state: AgentState):
  print('--extract_company_and_industry--entered--')
  text = state['messages'][-1].content 
  
  # Define patterns for extracting company and industry
  company_pattern = r'Company:\s*"([^"]+)"'
  industry_pattern = r'Industry:\s*"([^"]+)"'
  
  # Search for matches
  company_match = re.search(company_pattern, text)
  industry_match = re.search(industry_pattern, text)
  
  # Extract matched groups or return None if not found
  company_name = company_match.group(1) if company_match else None
  industry_name = industry_match.group(1) if industry_match else None
  queries = []
  if company_name:
      queries.extend([f'{company_name} Annual report pdf latest AND {company_name} website']) 
                      # f'{company_name} GenAI applications'])
                      # f'{company_name} key offerings and strategic focus areas (e.g., operations, supply chain, customer experience)', 
                      # f'{company_name} competitors and market share'])

  if industry_name:
      queries.extend([
        #  f'{industry_name} report latest mckinsey, deloitte, nexocode', 
                      f'{industry_name} GenAI applications'])
                      # f'{industry_name} trends, challenges and oppurtunities'])

  print('--extract_company_and_industry--finished--', queries)
  return {'queries': queries, 'company': company_name, 'industry': industry_name}


def web_scraping(state: AgentState):
  print('--web_scraping--entered--')
  queries = state['queries']
  link_list = []
  for query in queries:
      query_results = tavily_search.invoke({"query": query}) 
      link_list.extend(query_results)

  print('--web_scraping--finished--')
  return {'link_list': link_list}


# Agent Graph
def research_agent(user_query: str):
  builder = StateGraph(AgentState)
  builder.add_node('assistant', assistant)
  builder.add_node('names_extract', company_and_industry_query)
  builder.add_node('web_scraping', web_scraping)

  builder.add_edge(START, "assistant")
  builder.add_edge("assistant", "names_extract")
  builder.add_edge("names_extract", 'web_scraping')
  builder.add_edge("web_scraping", END)

  # memory
  memory = MemorySaver()
  react_graph = builder.compile(checkpointer=memory)
  
  config = {'configurable': {'thread_id':'1'}}
  messages = [HumanMessage(content=user_query)]
  agentstate_result = react_graph.invoke({'messages': messages}, config)
  
  return agentstate_result






