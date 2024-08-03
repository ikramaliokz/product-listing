import json
import os
import pinecone
from pinecone import ServerlessSpec
from utils import ImageFeatureExtractor
from utils import OpenAIEmbedder
import config


pinecone_api_key = os.environ.get('PINECONE_API_KEY') or ""
open_api_key = os.environ.get('OPENAI_API_KEY') or ""



class PineconeClient:
    def __init__(self, api_key, cloud='aws', region='us-east-1'):
        self.pc = pinecone.Pinecone(api_key=api_key)
        self.cloud = cloud
        self.region = region
        self.spec = ServerlessSpec(cloud=cloud, region=region)

    def get_index(self, index_name):
        return self.pc.Index(index_name)

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    


def get_pid_to_int_id(ids,intId_image_dict):
    pid_to_int_ID={}

    for id in ids:
        pid=str(intId_image_dict[id]['p_id'])
        if pid not in pid_to_int_ID:
            pid_to_int_ID[pid]=[id]
        else:
            pid_to_int_ID[pid].append(id)
    return pid_to_int_ID




    
# Load mappings
intId_image_data = load_json(config.int_id_to_image_data_path)

pid_to_text_data=load_json(config.pid_text_data_pth)

# intid_to_item_name = load_json('struct_jsons/titles-descrp-paths.json')

# Pinecone setup
pinecone_client = PineconeClient(api_key=pinecone_api_key)
product_listing_index = pinecone_client.get_index(config.product_index_name)
title_emb_index = pinecone_client.get_index(config.text_index_name)

# Initialize feature extractor and embedder
image_feature_extractor = ImageFeatureExtractor()
openai_embedder = OpenAIEmbedder()


def get_similar_product(image_pth, image_text): 

    # Query image
 
    # query_emb = image_feature_extractor.extract_features_lambda('product-listing-dataset/images/small/' + query_img_path)

    query_emb = image_feature_extractor.extract_features_lambda(image_pth)


    # Perform the image search
    search_result = product_listing_index.query(namespace="ns1", vector=query_emb, top_k=15, include_values=True)
    out_ids_img = {result['id']: round(result['score'], 4) for result in search_result['matches']}

    # Debugging: Print matching image IDs
    print('matching image ids', out_ids_img.keys())

    # Query title
    query_title = image_text
    query_title_emb = openai_embedder.embed([query_title])[0]

    # Perform the title search
    search_result_title = title_emb_index.query(namespace="ns1", vector=query_title_emb, top_k=15, include_values=True)
    out_pids_title = {result['id']: round(result['score'], 4) for result in search_result_title['matches']}

    # Debugging: Print matching text PIDs
    print('matching text Pids', out_pids_title.keys())
    
    # Get PID to internal ID mapping
    pid_to_int_ID = get_pid_to_int_id(out_ids_img, intId_image_data)

    # Debugging: Print matched PID to int-ids
    print("matched pid to int-ids", pid_to_int_ID.keys())

    # Ensure all keys are strings to avoid type mismatches
    pid_to_int_ID_keys = {str(key) for key in pid_to_int_ID.keys()}
    out_pids_title_keys = {str(key) for key in out_pids_title.keys()}

    # Calculate common PIDs
    common_pid=()
    common_pid = pid_to_int_ID_keys.intersection(out_pids_title_keys)

    if len(common_pid)<4:
        common_pid = list(common_pid)+list(pid_to_int_ID_keys.union(out_pids_title_keys)- common_pid)
        print('final pids', common_pid)

        out_paths=[]
        out_titles=[]
        for pid in common_pid[0:4]:
            out_paths.append(pid_to_text_data[pid]['image_paths'])
            out_titles.append([pid_to_text_data[pid]['item_name'] , pid_to_text_data[pid]['features']])
        
        return out_paths, out_titles




    # Debugging: Print common PIDs
    print('common pid', common_pid)


    # Filter out_pids_title using common_pid
    filtered_out_pids_title = {pid: scores for pid, scores in out_pids_title.items() if pid in common_pid}

    # Debugging: Print filtered out_pids_title and their keys
    print('filtered_out_pids_title', filtered_out_pids_title)
    print('filtered_out_pids_title keys', filtered_out_pids_title.keys())

    # Filter pid_to_int_ID using common_pid
    filtered_pid_to_int_ID = {pid: id for pid, id in pid_to_int_ID.items() if pid in common_pid}

    # Debugging: Print filtered_pid_to_int_ID
    print('filtered_pid_to_int_ID', filtered_pid_to_int_ID)

    # Verify that common_pid and filtered_out_pids_title keys are the same
    print('Are common_pid and filtered_out_pids_title keys the same?:', common_pid == set(filtered_out_pids_title.keys()))

    filtered_out_pids_title= {pid: scores for pid, scores in out_pids_title.items() if pid in common_pid}

    filtered_pid_to_int_ID={pid: id for pid, id in pid_to_int_ID.items() if pid in common_pid}

    
    
    
    weighted_averages=[]

    for pid in common_pid:
        for id in filtered_pid_to_int_ID[pid]:
            weight= (filtered_out_pids_title[pid]+ out_ids_img[id]) /2
            weighted_averages.append((pid,id,weight))

    weighted_averages = sorted(weighted_averages, key=lambda x: x[2], reverse=False)

    
    unique_pid = set()
    #   List to store the top unique k1 tuples
    top_unique_k1_results = []
    for k1, v1, value in weighted_averages:
        if k1 not in unique_pid:
            top_unique_k1_results.append((k1, v1, value))
            unique_pid.add(k1)

    print(top_unique_k1_results)
    print(unique_pid)

    out_paths=[]
    print('Number of matached products ', len(top_unique_k1_results))
    out_titles=[]
    for pid,id,val in top_unique_k1_results[0:4]:
        out_paths.append(pid_to_text_data[pid]['image_paths'])
        out_titles.append([pid_to_text_data[pid]['item_name'] , pid_to_text_data[pid]['features']])


    return out_paths, out_titles

def main():
   

    # Query image
    query_img_path = intId_image_data['0']['image_pth']

    query_title = pid_to_text_data['0']['item_name'] + pid_to_text_data['0']['features']

    out_paths,out_titles =get_similar_product(query_img_path,query_title)

    

    print("Retrieved paths:", out_paths)
    print("Retrieved titles:", out_titles)

if __name__ == "__main__":
    main()
