from setup import *
from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langgraph.constants import Send
from operator import add
from langgraph.graph import MessagesState
from typing import Annotated
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.jina import JinaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )


class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts to generate
    analysts: List[Analyst] # Analyst asking questions


class InterviewState(MessagesState):
  max_num_turns: int # Number turns of conversation
  context: Annotated[list, add] # Source docs
  analyst: Analyst # Analyst asking questions
  interview: str # Interview transcript
  sections: list # Final key we duplicate in outer state for Send() API


class SearchQuery(BaseModel):
  search_query: str = Field(None, description="Search query for retrieval.")



def create_analysts(state: GenerateAnalystsState):
    
  """ Create analysts """
  
  topic=state['topic']
  max_analysts=state['max_analysts']
      
  structured_llm = llm.with_structured_output(Perspectives)

  analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:
    1. First, review the research topic:{topic}
    2. Create {max_analysts} analysts with following roles:
      - Industry expert
      - GenAI expert
      - Business strategist
    3. Determine the most interesting themes based upon documents and/or feedback above.
    4. Pick the top {max_analysts} themes.
    5. For each theme, create one analyst with ALL of the following required fields:   - name: A fitting name for the analyst   - role: Their specific role or title   - affiliation: Their primary organization or institution   - description: A detailed description of their focus areas, concerns, and motives
    6. Ensure every analyst includes all four fields without exception. 
    Remember: Every analyst **MUST** have all four fields (name, role, affiliation, and description) properly defined. Incomplete personas are not acceptable."""

  # System message
  system_message = analyst_instructions.format(topic=topic, max_analysts=max_analysts)
  
  analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])

  # Write the list of analysis to state
  return {"analysts": analysts.analysts}




def vectorstore_writing(doc_splits):
    global retriever
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding = JinaEmbeddings(model_name='jina-embeddings-v3'),
        persist_directory='./chroma_db'
    )
    retriever = vectorstore.as_retriever()





def generate_question(state:InterviewState):
  """ Generate questions for the interview """
  
  # print('----STATE----', state)
  # Get the analyst persona
  analyst = state['analyst']
  messages = state['messages']
  context = state["context"]

  question_instructions = """You are an analyst tasked with interviewing an expert to learn about the use of Generative AI (GenAI) applications in a specific industry or company, if mentioned.

    Your goal is to uncover interesting and specific insights related to the topic of Generative AI use cases.

    Interesting: Insights that are surprising, non-obvious, or reveal unique applications of GenAI in the industry or company.
    Specific: Insights that avoid generalities and include specific examples or case studies relevant to the company’s offerings, strategic focus areas, or the industry’s needs.
    Focus Areas:
    Explore the company's key offerings and strategic focus areas (e.g., operations, supply chain, customer experience, etc.), if the company is named.
    Discuss industry-wide trends, innovations, and opportunities enabled by GenAI, such as improved operational efficiency, enhanced customer experiences, or streamlined supply chain processes.
    Gather details on the company or industry's vision and products, focusing on how GenAI can be applied to enhance or transform their workflows.
    Task:
    Begin by introducing yourself with a name that fits your persona, then ask your question.

    Continue asking follow-up questions to drill down into:

    Specific GenAI use cases within the company's domain or the industry.
    How these applications align with the company's or industry's strategic goals.
    Real-world examples or future opportunities for integrating GenAI into their processes.
    Complete the interview by saying:
    "Thank you so much for your help!"

    Remember to stay in character throughout the conversation, reflecting your persona and the provided goals."""

  # Generate the question
  question = llm.invoke([SystemMessage(content=question_instructions)]+[HumanMessage(content="Generate the question.")])
  
  return {"messages": [question]}



def search_vectorstore(state: InterviewState):
    
    """ Retrieve docs from Docstore """
    
    # Search query writing
    search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

    Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
            
    First, analyze the full conversation.

    Pay particular attention to the final question posed by the analyst.

    Convert this final question into a well-structured web search query""")

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])
    
    # Search
    search_docs = retriever.invoke(input=search_query.search_query)

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 



def generate_answer(state: InterviewState):
    
    """ Node to answer a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]


    answer_instructions = """You are an expert being interviewed by an analyst.

    Here is analyst area of focus: {goals}. 
            
    You goal is to answer a question posed by the interviewer.

    To answer question, use this context:
            
    {context}

    When answering questions, follow these guidelines:
            
    1. Use only the information provided in the context. 
            
    2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

    3. The context contain sources at the topic of each individual document.

    4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

    5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
            
    6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
            
    [1] assistant/docs/llama3_1.pdf, page 7 
            
    And skip the addition of the brackets as well as the Document source preamble in your citation."""



    # Answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)
            
    # Name the message as coming from the expert
    answer.name = "expert"
    
    # Append it to state
    return {"messages": [answer]}


def save_interview(state: InterviewState):
    
    """ Save interviews """

    # Get messages
    messages = state["messages"]
    
    # Convert interview to a string
    interview = get_buffer_string(messages)
    
    # Save to interviews key
    return {"interview": interview}



def route_messages(state: InterviewState, 
                   name: str = "expert"):

    """ Route between question and answer """
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers 
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"



def write_section(state: InterviewState):

    """ Node to answer a question """

    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
   

    section_writer_instructions = """You are an expert technical writer. 
            
    Your task is to create a short, easily digestible section of a report based on a set of source documents.

    1. Analyze the content of the source documents: 
    - The name of each source document is at the start of the document, with the <Document tag.
            
    2. Create a report structure using markdown formatting:
    - Use ## for the section title
    - Use ### for sub-section headers
            
    3. Write the report following this structure:
    a. Title (## header)
    b. Summary (### header)
    c. Sources (### header)

    4. Make your title engaging based upon the focus area of the analyst: 
    {focus}

    5. For the summary section:
    - Set up summary with general background / context related to the focus area of the analyst
    - Emphasize what is novel, interesting, or surprising about insights gathered from the interview
    - Create a numbered list of source documents, as you use them
    - Do not mention the names of interviewers or experts
    - Aim for approximately 400 words maximum
    - Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
            
    6. In the Sources section:
    - Include all sources used in your report
    - Provide full links to relevant websites or specific document paths
    - Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
    - It will look like:

    ### Sources
    [1] Link or Document name
    [2] Link or Document name

    7. Be sure to combine sources. For example this is not correct:

    [3] https://ai.meta.com/blog/meta-llama-3-1/
    [4] https://ai.meta.com/blog/meta-llama-3-1/

    There should be no redundant sources. It should simply be:

    [3] https://ai.meta.com/blog/meta-llama-3-1/
            
    8. Final review:
    - Ensure the report follows the required structure
    - Include no preamble before the title of the report
    - Check that all guidelines have been followed"""


    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
                
    # Append it to state
    return {"sections": [section.content]}



# Add nodes and edges 
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_rag", search_vectorstore)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_rag")
interview_builder.add_edge("search_rag", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# Interview 
memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")




class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report
    human_analyst_feedback: Optional[str] # Human feedback



def initiate_all_interviews(state: ResearchGraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """    

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback')
    if human_analyst_feedback:
        # Return to create_analysts
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        return [Send("conduct_interview", {"analyst": analyst,
                                           "messages": [HumanMessage(
                                               content=f"So you said you were writing an article on {topic}?")], 
                                                       }) for analyst in state["analysts"]]

report_writer_instructions = '''You are a technical writer tasked with creating a report on the overall topic:  

**{topic}**  

Your team of analysts has conducted interviews and written memos based on their findings. Your task is to consolidate the insights from these memos into a cohesive and structured report, following this format:  

Think deeply and Generate atleat 2 use cases based on the memos.

### Format for Each Use Case  
1. **Title Header:** Use a descriptive title for each use case, such as "## Use Case 1: AI-Powered Predictive Maintenance."  
2. **Objective/Use Case:** Summarize the primary goal or application of AI for this use case in one or two sentences.  
3. **AI Application:** Describe the specific AI technologies or methods used to achieve the objective.  
4. **Cross-Functional Benefit:** Outline the key benefits across various functions, formatted as bullet points, specifying which department or area benefits from the AI use case.  

### Example Format:  

## Use Case 1: AI-Powered Predictive Maintenance  
**Objective/Use Case:** Reduce equipment downtime and maintenance costs by predicting equipment failures before they occur.  
**AI Application:** Implement machine learning algorithms that analyze real-time sensor data from machinery to predict potential failures and schedule maintenance proactively.  
**Cross-Functional Benefit:**  
- **Operations & Maintenance:** Minimizes unplanned downtime and extends equipment lifespan.  
- **Finance:** Reduces maintenance costs and improves budgeting accuracy.  
- **Supply Chain:** Optimizes spare parts inventory based on predictive insights.  

## Use Case 2: Real-Time Quality Control with Computer Vision  
**Objective/Use Case:** Enhance product quality by detecting defects in products during manufacturing.  
**AI Application:** Deploy AI-powered computer vision systems on production lines to identify surface defects and inconsistencies in real time.  
**Cross-Functional Benefit:**  
- **Quality Assurance:** Improves defect detection accuracy and reduces scrap rates.  
- **Production:** Enables immediate corrective actions, enhancing overall efficiency.  
- **Customer Satisfaction:** Delivers higher-quality products, strengthening client relationships.  

### Report Guidelines  
1. Begin with the first use case title in the specified format.  
2. Do not include any preamble or introductory text for the report.  
3. Consolidate insights into distinct use cases, with a focus on clarity and relevance.  
4. Preserve any citations included in the memos, formatted in brackets, e.g., [1], [2].  
5. After detailing all use cases, include a **Sources** section with the title: `## Sources`.  
6. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:
[3] https://ai.meta.com/blog/meta-llama-3-1/ 

### Your Inputs  
You will be given a collection of memos from your analysts under `{context}`. Extract and distill insights into specific use cases, ensuring each use case adheres to the prescribed format.''' 

def write_report(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")]) 
    return {"content": report.content}


def human_feedback(state: ResearchGraphState):
    """ No-op node that should be interrupted on """
    pass



def write_introduction(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

    You will be given all of the sections of the report.

    You job is to write a crisp and compelling introduction or conclusion section.

    The user will instruct you whether to write the introduction or conclusion.

    Include no pre-amble for either section.

    Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

    Use markdown formatting. 

    For your introduction, create a compelling title and use the # header for the title.

    For your introduction, use ## Introduction as the section header. 

    For your conclusion, use ## Conclusion as the section header.

    Here are the sections to reflect on for writing: {formatted_str_sections}"""


    # Summarize the sections into a final report
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

    You will be given all of the sections of the report.

    You job is to write a crisp and compelling introduction or conclusion section.

    The user will instruct you whether to write the introduction or conclusion.

    Include no pre-amble for either section.

    Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

    Use markdown formatting. 

    For your introduction, create a compelling title and use the # header for the title.

    For your introduction, use ## Introduction as the section header. 

    For your conclusion, use ## Conclusion as the section header.

    Here are the sections to reflect on for writing: {formatted_str_sections}"""


    # Summarize the sections into a final report
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}



def usecase_agent_func(topic,max_analysts):
  # Add nodes and edges 
  builder = StateGraph(ResearchGraphState)
  builder.add_node("create_analysts", create_analysts)
  builder.add_node("human_feedback", human_feedback)
  builder.add_node("conduct_interview", interview_builder.compile())
  builder.add_node("write_report",write_report)
  builder.add_node("write_introduction",write_introduction)
  builder.add_node("write_conclusion",write_conclusion)
  builder.add_node("finalize_report",finalize_report)

  # Logic
  builder.add_edge(START, "create_analysts")
  builder.add_edge("create_analysts", "human_feedback")
  builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
  builder.add_edge("conduct_interview", "write_report")
  builder.add_edge("conduct_interview", "write_introduction")
  builder.add_edge("conduct_interview", "write_conclusion")
  builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
  builder.add_edge("finalize_report", END)

  # Compile
  memory = MemorySaver()
  graph = builder.compile(checkpointer=memory)
  config = {"configurable": {"thread_id": "1"}}
  graph.invoke({"topic":topic,
                "max_analysts":max_analysts,
                'human_analyst_feedback': None}, 
                 config)
  final_state = graph.get_state(config)
  report = final_state.values.get('final_report')

  print('-----REPORT-----', report)

  return report



