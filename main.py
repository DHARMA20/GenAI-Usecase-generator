from setup import *
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font

from agents import research_agent
from vectorstore import extract_urls, urls_classify_list, clean_and_extract_html_data, pdf_extraction
from usecase_agent import usecase_agent_func, vectorstore_writing
from feasibility_agent import feasibility_agent_func

def create_excel(output_list):

  df = pd.DataFrame(output_list)

  # Create an Excel workbook
  wb = Workbook()
  ws = wb.active
  ws.title = "Use Cases"

  # Write headers
  headers = ['Use Case', 'Description', 'URLs']
  ws.append(headers)

  # Write data rows
  for index, row in df.iterrows():
      # Add use_case and description
      ws.cell(row=index + 2, column=1).value = row['use_case']
      ws.cell(row=index + 2, column=2).value = row['description']
      
      # Combine URLs into a single string separated by newlines
      urls_combined = '\n'.join(row['urls_list'])
      cell = ws.cell(row=index + 2, column=3)
      cell.value = urls_combined

      # Format hyperlinks individually for readability
      for i, url in enumerate(row['urls_list']):
          cell.hyperlink = url  # Hyperlinks are applied per cell for each link
          cell.font = Font(color="0000FF", underline="single")

  # Save the Excel file
  wb.save("GenAI_use_cases_feasibility.xlsx")


# Research Agent
agentstate_result = research_agent('GenAI applications in MRF Tyres')


# Vector Store
urls = extract_urls(agentstate_result)
pdf_urls, html_urls = urls_classify_list(urls)
html_docs = clean_and_extract_html_data(html_urls)
pdf_docs = pdf_extraction(pdf_urls)

doc_splits = html_docs.__add__(pdf_docs)

vectorstore_writing(doc_splits)


# Use-case agent
company_name = agentstate_result['company']
industry_name = agentstate_result['industry']

if company_name:
  topic = f'GenAI Usecases in {company_name} and {industry_name} industry. Explore {company_name} GenAI applications, key offerings, strategic focus areas, competitors, and market share.'
topic = f'GenAI Usecases in {industry_name}. Explore {industry_name} GenAI applications, trends, challenges, and opportunities.'
max_analysts = 3 

report = usecase_agent_func(topic, max_analysts)



# File name for the Markdown file
file_name = "report.md"

# Write the content to the file
with open(file_name, "w") as md_file:
    md_file.write(report)

print(f"Markdown file '{file_name}' has been created.")


pd_dict = feasibility_agent_func(report)
create_excel(pd_dict)