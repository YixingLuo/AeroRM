import pandas as pd
import re


models = ['glm4']
for model in models:
    file_path = model + '_results.xlsx'  
    df = pd.read_excel(file_path)

    keyword_pattern = re.compile(r'<keyword>(.*?)</keyword>', re.IGNORECASE)
    text_description_pattern = re.compile(r'<description>(.*?)</description>', re.DOTALL | re.IGNORECASE)
    domain_pattern = re.compile(r'<domain>(.*?)</domain>', re.IGNORECASE)
    id_pattern = re.compile(r'<id>(.*?)</id>', re.IGNORECASE)
    name_pattern = re.compile(r'<name>(.*?)</name>', re.IGNORECASE)


    extracted_data = []

    for index, row in df.iterrows():
        knowledge_model = row['Knowledge Model'] 
        
        keywords = re.findall(keyword_pattern, knowledge_model)
        text_descriptions = re.findall(text_description_pattern, knowledge_model)
        domains = re.findall(domain_pattern, knowledge_model)
        ids = re.findall(id_pattern, knowledge_model)
        names = re.findall(name_pattern, knowledge_model)

        
        extracted_data.append({
            'Requirement': row['Requirement'],
            'Code': row['Code'],
            'Device': row['Device'],
            'Knowledge Model': row['Knowledge Model'],
            'id': ids if ids else None,  
            'name': names if names else None, 
            'keyword': keywords if keywords else None,
            'domain': domains if domains else None,
            'description': text_descriptions if text_descriptions else None
        })


    extracted_df = pd.DataFrame(extracted_data)
    output_file = model + 'EmbedIP.xlsx'  
    extracted_df.to_excel(output_file, index=False)

    print(f"The extracted data has been saved to {output_file}")
