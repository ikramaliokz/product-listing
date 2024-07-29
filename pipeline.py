import json
import os
import pinecone
from pinecone import ServerlessSpec
from utils import ImageFeatureExtractor
from utils import OpenAIEmbedder


pinecone_api_key = os.environ.get('PINECONE_API_KEY') or ""
open_api_key = os.environ.get('OPENAI_API_KEY') or ""

# openai doesn't need to be initialized, but need to set api key
# os.environ["OPENAI_API_KEY"] = ""

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
intId_image_data = load_json('updated_struct_jsons/intid_to_imagepth_pid.json')

pid_to_text_data=load_json('updated_struct_jsons/titles-descrp-paths.json')

# intid_to_item_name = load_json('struct_jsons/titles-descrp-paths.json')

# Pinecone setup
pinecone_client = PineconeClient(api_key=pinecone_api_key)
product_listing_index = pinecone_client.get_index('full-pd-embeddings')
title_emb_index = pinecone_client.get_index('full-text-embeddings')

# Initialize feature extractor and embedder
image_feature_extractor = ImageFeatureExtractor()
openai_embedder = OpenAIEmbedder()


def get_similar_product(image_pth, image_text): 

    # Query image
    query_img_path = image_pth
    query_emb = image_feature_extractor.extract_features('product-listing-dataset/images/small/' + query_img_path).tolist()

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
    common_pid = pid_to_int_ID_keys.intersection(out_pids_title_keys)

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
# List to store the top unique k1 tuples
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

























    # print('matching image pids', out_pid_image)

    # out_pid_image_vals=dict(zip(out_pid_image,out_ids_img.values()))

    # pId_int_id=dict(zip(out_ids_img, out_pid_image))

    

    # # Find common pairs and calculate weighted averages
    # common_pid=set(out_pid_image_vals.keys()).intersection(out_ids_title.keys())

    # print(common_pid)
    # if len(common_pid)!=0:
    #     common_pairs = {key: (out_pid_image_vals[key], out_ids_title[key]) for key in common_pid}
    #     weighted_averages = {key: (value[0] + value[1]) / 2 for key, value in common_pairs.items()}
    # else:
    #     common_pairs=out_ids_title
    #     weighted_averages=out_ids_title

    
    # sorted_weighted_averages = sorted(weighted_averages.items(), key=lambda item: item[1])
    # if len(sorted_weighted_averages) > 4:
    #     top_4_keys = [item[0] for item in sorted_weighted_averages[:4]]
    # else:
    #     top_4_keys = [item[0] for item in sorted_weighted_averages]
    
    # # Retrieve paths and titles
    # out_paths = [intId_image_data[pId_int_id[key]] for key in top_4_keys]
    # out_titles = [[pid_to_text_data[key]['item_name'] , pid_to_text_data[key]['features'] ] for key in top_4_keys]

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
