from setup import *
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from typing import List
from langchain_community.tools import TavilySearchResults


keyword_search = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)


# Define the UseCaseKeywords model to include use_case, description, and keyword
class UseCaseKeywords(BaseModel):
    use_case: str
    description: str
    keyword: str

    # Override the dict method to return a dictionary with use_case as the key
    def to_dict(self) -> dict:
        return {
            'use_case': self.use_case,
            'description': self.description,
            'keyword': self.keyword
        }

# Define the KeywordGenerationResponse model to contain a list of UseCaseKeywords
class KeywordGenerationResponse(BaseModel):
    data: List[UseCaseKeywords]

    # Convert the list of UseCaseKeywords to a list of dictionaries
    def to_list_of_dicts(self) -> List[dict]:
        return [entry.to_dict() for entry in self.data]


def keyword_generation(report):

  query_generation_sys_prompt = SystemMessage(content='''You are an expert in creating precise and relevant keyword queries to search for datasets. Your task is to generate a keyword query for each use case provided below. These queries should be optimized for searching datasets on platforms such as GitHub, Kaggle, and Hugging Face.  

  **Instructions:**  
  1. Extract the key concepts from the use case (e.g., objectives, AI application, and domain).  
  2. Formulate a concise, descriptive query using relevant terms and synonyms.  
  3. Include terms related to data types (e.g., "customer data," "chat logs," "shopping behavior"), AI techniques (e.g., "machine learning," "recommendation systems"), and target domain (e.g., "e-commerce," "retail").  
  4. Create a output dictionary with the use case title as the key and the keyword query as the value.

  **Use Cases: Examples** 
  ## Use Case 1: Personalized Shopping Experiences with GenAI  
  **Objective/Use Case:** Create tailored shopping experiences for individual customers based on their browsing history, purchasing behavior, and preferences.  
  **AI Application:** Implement machine learning algorithms that analyze customer data to generate personalized offers, marketing communications, and product recommendations.  
  **Cross-Functional Benefit:**  
  - **Marketing:** Increases customer satisfaction and loyalty through targeted marketing efforts.  
  - **Sales:** Boosts sales by offering relevant products to customers.  
  - **Customer Service:** Enhances customer experience through personalized support.  

  ## Use Case 2: AI-Powered Chatbots for Customer Service  
  **Objective/Use Case:** Improve in-store customer service by providing instant assistance and directing customers to relevant products.  
  **AI Application:** Develop GenAI-powered chatbots that analyze customer queries and provide accurate responses, suggesting related products and services.  
  **Cross-Functional Benefit:**  
  - **Customer Service:** Reduces wait times and improves customer satisfaction.  
  - **Sales:** Increases sales by suggesting relevant products to customers.  
  - **Operations:** Enhances employee productivity by automating routine tasks.  

  Example output:
    [{'use_case' : "Personalized Shopping Experiences with GenAI" , 
      'description':"AI-driven personalization enhances customer satisfaction through tailored offers, recommendations, and marketing based on individual preferences."                                
    'keyword': "e-commerce personalized shopping data customer behavior recommendation system offers dataset"},
    {'use_case': "AI-Powered Chatbots for Customer Service" , 
      'description': AI chatbots provide instant, accurate assistance, improving customer service, increasing sales, and boosting operational efficiency.                                
    'keyword': "customer service chatbot dataset customer queries retail e-commerce AI automation"}]''')


  # Example usage (you will use llm to generate the output)
  Keyword_generation_llm = llm.with_structured_output(KeywordGenerationResponse)

  # Your report as input (ensure that this variable is properly formatted and available)
  report_msg = HumanMessage(content=f'The usecases are as follows {report}')

  # Invoke the LLM and get the response
  output_dict = Keyword_generation_llm.invoke([query_generation_sys_prompt, report_msg])

  # Convert the response to a list of dictionaries
  output_list = output_dict.to_list_of_dicts()

  return output_list



def dataset_search(output_list):
  for usecase_dict in output_list:
    query = usecase_dict['keyword']
    query_format = 'kaggle OR github OR huggingface ' + 'AND' + query
    links = keyword_search.invoke({'query': query_format})
    usecase_dict['links'] = links
  return output_list



def grouping_urls(output_list):
  for dict_item in output_list:
    urls_list = []
    for ele in dict_item['links']:
      urls_list.append(ele['url'])
    dict_item['urls_list'] = urls_list
  return output_list



def delete_columns(output_list):
  # Specify the keys you want to include
  keys_to_del = ['links', 'keyword']

  for dict_item in output_list:
    for key in keys_to_del:
      del dict_item[key]
  return output_list


def feasibility_agent_func(report):
  dict_list = keyword_generation(report)
  dict_links = dataset_search(dict_list)
  urls_dict = grouping_urls(dict_links)
  pd_dict = delete_columns(urls_dict)

  return pd_dict