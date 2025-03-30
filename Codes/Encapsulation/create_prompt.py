from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from http import HTTPStatus
import dashscope
import random
from openai import OpenAI

os.environ["ZHIPUAI_API_KEY"] = ""#your key

file_path = "../../Sun Search Control System/RCD.xlsx" 
df = pd.read_excel(file_path)


examples_path = "example.txt" 
with open(examples_path, 'r', encoding='utf-8') as file:
    lines = [line.strip() for line in file]
examples = '\n'.join(lines)


def get_implement(IP_name, IP_path='./SAMCode_IPSynthesis'):
    IP_folder = os.path.join(IP_path, IP_name)
    IP_folder_implement =  os.path.join(IP_folder, 'Implement')
    c_files = []
    h_files = []
    for root, dirs, files in os.walk(IP_folder_implement):
        for file in files:
            if file.endswith(".c"):
                c_files.append(os.path.join(root, file))
            if file.endswith(".h"):
                h_files.append(os.path.join(root, file))
    for item in c_files:   
        with open(item, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
    c_file = '\n'.join(lines)  
    for item in h_files:   
        with open(item, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
    h_file = '\n'.join(lines)   

    return h_file + ('\n') + c_file    

questions = []
for index, row in df.iterrows():
    IP_Req = row['Requirement']
    IP_Code = row['Code']
    IP_Device = row['Device']
    questions.append((IP_Req, IP_Code, IP_Device))

def call_stream_with_messages(query):

    client = OpenAI(
        api_key="", # your key
        base_url="https://open.bigmodel.cn/api/paas/v4"
    )
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": query}
    ]
    completion = client.chat.completions.create(
        model="glme-9b-chat",
        messages=messages,
        max_tokens=3000,
        temperature=0,
        timeout=60
    )
    response_content = completion.choices[0].message.content

    return response_content

def batch_generate_and_compare_responses(questions, max_retries=3, retry_delay=5):
    results = []
    for IP_Req, IP_Code, IP_Device in tqdm(questions, desc="Processing questions"):
        ## type 1 zero-shot:
        IP_input = f"""
## Task Description
You are a software asset modeling expert in the embedded software domain. Your task is to generate the knowledge model in XML format based on the software asset information provided. Follow the knowledge abstraction guidelines and the given example to structure the model.
## Knowledge Abstraction Guidelines 
- id: a unique identification of IP, from which the device name, application domain, functionality, and other relevant information can be extracted.
- name: a noun phrase to directly reflect the functionality it is designed to achieve and the physical device involved.
- keyword: terms that should effectively capture the characteristics of the IP and must be drawn from the domain-specific terminology dictionary.
- domain: the application domain specifies the fields in which \IP is suitable for use, and this description must also be based on the domain-specific terminology dictionary.
- description: the functionality description of the IP, derived from its requirement, code snippet, and device information, should remain concise, avoiding the use of invented or semantically ambiguous terms.
## Input
- **Requirements Description**:
```text
{IP_Req}
```
- **Code Implementation**:
```C
{IP_Code}
```
- **Device Information**:
```text
{IP_Device}
```
"""
        input_content = IP_input + '\n' + examples
        generated_answer = ''
        messages = [
            # SystemMessage(content=system_content),
            HumanMessage(content=input_content),
        ]
        retries = 0
        while retries < max_retries:
            try:
                # response = chat_model.invoke(messages)
                # generated_answer = response.content
                generated_answer = call_stream_with_messages(input_content)
                break
            except Exception as e:
                print(f"Error: {str(e)}. Retrying ({retries+1}/{max_retries})...")
                time.sleep(retry_delay)
                retries += 1
                generated_answer = f"Error after {max_retries} retries: {str(e)}"
        
        results.append((IP_Req, IP_Code, IP_Device, input_content, generated_answer))
    return results

results = batch_generate_and_compare_responses(questions)

df = pd.DataFrame(results)
df.columns = ['Requirement', 'Code', 'Device', 'Prompt', 'Knowledge Model']
output_xlsx_path = "glm4_results.xlsx"
df.to_excel(output_xlsx_path, index=False)
