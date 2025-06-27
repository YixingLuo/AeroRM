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
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ------------- 1. Environment & Path Setup -------------
os.environ["ZHIPUAI_API_KEY"] = ""  # Your ZhipuAI API key, if needed

file_path = "../../Sun Search Control System/RCD.xlsx"   # Path to your Excel file with requirements
examples_path = "example.txt"                             # Path to your example file

# ------------- 2. Load Requirements and Examples -------------
df = pd.read_excel(file_path)
with open(examples_path, 'r', encoding='utf-8') as file:
    lines = [line.strip() for line in file]
examples = '\n'.join(lines)

questions = []
for index, row in df.iterrows():
    IP_Req = row['Requirement']
    IP_Device = row['Device']
    questions.append((IP_Req, IP_Device))

# ------------- 3. Load FAISS Index and Embedding Model -------------
# Make sure faiss_index and faiss_texts are aligned: each text's vector is in the same order as the texts list
faiss_index_path = "your_faiss.index"          # Path to your FAISS index file
faiss_texts_path = "your_faiss_texts.txt"      # Path to your text file, one text per line

faiss_index = faiss.read_index(faiss_index_path)
with open(faiss_texts_path, 'r', encoding='utf-8') as f:
    faiss_texts = [line.strip() for line in f]

# Initialize a sentence-transformers embedding model; change the model if needed
model = SentenceTransformer('BAAI/bge-base-zh')  # You may use 'all-MiniLM-L6-v2' for English
def embed_fn(texts):
    """
    Convert a list of texts into embeddings using sentence-transformers.
    """
    return model.encode(texts, normalize_embeddings=True)

# ------------- 4. FAISS Retrieval Function -------------
def retrieve_top_k_chunks(query, faiss_index, faiss_texts, embed_fn, k=3):
    """
    Retrieve the top-k most similar text chunks from FAISS given a query.
    """
    query_vec = embed_fn([query]).astype('float32')
    D, I = faiss_index.search(query_vec, k)
    top_chunks = [faiss_texts[i] for i in I[0]]
    return top_chunks

# ------------- 5. Chat API Call Function (GLM-4 or OpenAI Compatible) -------------
def call_stream_with_messages(query):
    """
    Send a prompt to the chat model (e.g., GLM-4) and return the generated response.
    """
    client = OpenAI(
        api_key="",  # Your API KEY here
        base_url="https://open.bigmodel.cn/api/paas/v4"  # GLM4 endpoint, or change to OpenAI's endpoint
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

# ------------- 6. Batch Processing Pipeline -------------
def batch_generate_and_compare_responses(questions, faiss_index, faiss_texts, embed_fn, max_retries=3, retry_delay=5):
    """
    For each (requirement, device) pair:
    - Retrieve similar text chunks via FAISS.
    - Compose the prompt including retrieved context.
    - Call the chat API to generate a response.
    - Collect all results.
    """
    results = []
    for IP_Req, IP_Device in tqdm(questions, desc="Processing questions"):
        # Step 1: Retrieve most relevant context from FAISS
        retrieved_chunks = retrieve_top_k_chunks(IP_Req, faiss_index, faiss_texts, embed_fn, k=3)
        retrieved_chunks_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(retrieved_chunks)])

        # Step 2: Build the prompt
        IP_input = f"""
        ## Instruction
        You are tasked with generating a structured knowledge encapsulation of reusable requirements artifacts based on provided context-specific criteria. The five key features to detail are: ID, Name, Keyword, Domain, and Description.
        ## Guidelines to Follow 
        - id: a unique identifier of the AeroR, incorporating relevant device name, function, and application domain. 
        - name: a noun phrase summarizing the functionality and device involved. 
        - keyword: 2–5 key terms reflecting the essential technical elements. 
        - domain: the applicable operational domain (e.g., attitude control, telemetry). 
        - description: a concise summary derived from the requirement and device context—avoiding hallucinated or vague terms.
        ## Retrieved Context
        {retrieved_chunks_str}
        ## Input
        - **Requirements Description**:
        ```text
        {IP_Req}
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
        
        results.append((IP_Req, IP_Device, input_content, generated_answer))
    return results

results = batch_generate_and_compare_responses(questions)

df = pd.DataFrame(results)
df.columns = ['Requirement', 'Device', 'Prompt', 'Knowledge Model']
output_xlsx_path = "glm4_results.xlsx"
df.to_excel(output_xlsx_path, index=False)
