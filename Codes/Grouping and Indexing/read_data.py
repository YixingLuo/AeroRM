import pandas as pd

def read_data(file_path):
    """
    读取 Excel 文件并返回 DataFrame。
    :param file_path: Excel 文件路径
    :return: DataFrame
    """
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None