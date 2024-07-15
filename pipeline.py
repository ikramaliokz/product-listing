import os
from pinecone import ServerlessSpec, Pinecone
from utils import extract_features, embed
import json


pinecone_api_key = os.environ.get('PINECONE_API_KEY') or "de307987-1233-4e50-b76f-5caa45515a64"



# openai doesn't need to be initialized, but need to set api key
os.environ["OPENAI_API_KEY"] = "sk-proj-W5AbObl8rTot2sToNh3AT3BlbkFJ3ILh68KBofwVQIE1jtRM"

with open('img_paths_to_int_id.json', 'r') as f1:

    img_paths_to_int_id = json.load(f1)

with open('intid_to_item_name.json', 'r') as f2:

    intid_to_item_name = json.load(f2)

INDEX_NAME = 'pd-embeddings'
INDEX_NAME_2 = 'titles-embeddings'

pc = Pinecone(api_key=pinecone_api_key)

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)
product_listing_index = pc.Index(INDEX_NAME)
title_emb_index = pc.Index(INDEX_NAME_2)


query_img_path = img_paths_to_int_id['0']
query_title = intid_to_item_name['0']

query_emb = extract_features('images/small/'+query_img_path).tolist()
print(len(query_emb))
# now query
search_result = product_listing_index.query(namespace="ns1", vector=query_emb, top_k=15, include_values=True)
out_ids_img = {}
for result in search_result['matches']:
    out_ids_img[result['id']] = round(result['score'], 4)


# now query
query_title_emb = embed(query_title)[0]
search_result_title = title_emb_index.query(namespace="ns1", vector=query_title_emb, top_k=15, include_values=True)
out_ids_title = {}
for result in search_result_title['matches']:
    out_ids_title[result['id']] = round(result['score'], 4)

    # print(result)
# Find the intersection of keys and get the corresponding key-value pairs from both dictionaries
common_pairs = {key: (out_ids_img[key], out_ids_title[key]) for key in out_ids_img.keys() & out_ids_title.keys()}


# Calculate the weighted average for each key
weighted_averages = {key: (value[0] + value[1]) / 2 for key, value in common_pairs.items()}

# Convert the dictionary to a list of tuples and sort it by the weighted average
sorted_weighted_averages = sorted(weighted_averages.items(), key=lambda item: item[1])

# Get the top 4 keys based on the sorted weighted averages
top_4_keys = [item[0] for item in sorted_weighted_averages[:4]]

# Print the top 4 keys
print("top 4 keys: ",top_4_keys)
out_paths = []
out_titles = []
for key in top_4_keys:
    out_paths.append(img_paths_to_int_id[key])
    out_titles.append(intid_to_item_name[key])

print("Retrievved paths:", out_paths)


