def extract_cluster_data_first(cluster_indices, df):
    """
    Extract the `code`, `name`, `Keywords`, and `Domains` fields from the clusters.
    Assuming the DataFrame contains the corresponding columns.
    """
    cluster_data = []
    for idx in cluster_indices:
        Requirements = df.iloc[idx]['Requirements']
        name = df.iloc[idx]['name']
        keywords = df.iloc[idx]['Keywords']
        domains = df.iloc[idx]['Domains']
        cluster_data.append({
            'Requirements': Requirements,
            'name': name,
            'Keywords': keywords,
            'Domains': domains
        })
    return cluster_data

def extract_cluster_data_recursion(cluster_indices, df, recursion_level):
    """
    Extract the `cluster_name` and `cluster_description` fields from the clusters.
    Assuming the DataFrame contains the corresponding columns, with suffixes indicating recursion levels.
    """
    cluster_name_column = f'cluster_name_{recursion_level}'
    cluster_description_column = f'cluster_description_{recursion_level}'

    if cluster_name_column not in df.columns or cluster_description_column not in df.columns:
        raise ValueError(f"Error: Missing '{cluster_name_column}' or '{cluster_description_column}' in the DataFrame.")

    cluster_data = []
    for idx in cluster_indices:
        cluster_name = df.iloc[idx][cluster_name_column]
        cluster_description = df.iloc[idx][cluster_description_column]

        if cluster_description is None:
            cluster_description = ""

        cluster_data.append({
            'cluster_name': cluster_name,
            'cluster_description': cluster_description
        })
    return cluster_data

def extract_cluster_data_recursion_pre(cluster_indices, df):
    """
    Extract the `cluster_name` and `cluster_description` fields from the clusters.
    Assuming the DataFrame contains the corresponding columns.
    """
    cluster_data = []
    for idx in cluster_indices:
        cluster_name = df.iloc[idx]['cluster_name']
        cluster_description = df.iloc[idx]['cluster_description']
        cluster_data.append({
            'cluster_name': cluster_name,
            'cluster_description': cluster_description
        })
    return cluster_data