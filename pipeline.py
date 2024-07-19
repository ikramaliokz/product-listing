import json
import os
import pinecone
from pinecone import ServerlessSpec
from utils import ImageFeatureExtractor
from utils import OpenAIEmbedder


pinecone_api_key = os.environ.get('PINECONE_API_KEY') or ""

# openai doesn't need to be initialized, but need to set api key
os.environ["OPENAI_API_KEY"] = ""

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

def main():
    # Load mappings
    img_paths_to_int_id = load_json('img_paths_to_int_id.json')
    intid_to_item_name = load_json('intid_to_item_name_and_features.json')

    # Pinecone setup
    pinecone_client = PineconeClient(api_key=pinecone_api_key)
    product_listing_index = pinecone_client.get_index('pd-embeddings')
    title_emb_index = pinecone_client.get_index('text-embeddings')

    # Initialize feature extractor and embedder
    image_feature_extractor = ImageFeatureExtractor()
    openai_embedder = OpenAIEmbedder()

    # Query image
    query_img_path = img_paths_to_int_id['0']
    query_emb = image_feature_extractor.extract_features('images/small/' + query_img_path).tolist()
    search_result = product_listing_index.query(namespace="ns1", vector=query_emb, top_k=15, include_values=True)
    out_ids_img = {result['id']: round(result['score'], 4) for result in search_result['matches']}

    # Query title
    query_title = intid_to_item_name['0']
    query_title_emb = openai_embedder.embed([query_title])[0]
    search_result_title = title_emb_index.query(namespace="ns1", vector=query_title_emb, top_k=15, include_values=True)
    out_ids_title = {result['id']: round(result['score'], 4) for result in search_result_title['matches']}

    # Find common pairs and calculate weighted averages
    common_pairs = {key: (out_ids_img[key], out_ids_title[key]) for key in out_ids_img.keys() & out_ids_title.keys()}
    weighted_averages = {key: (value[0] + value[1]) / 2 for key, value in common_pairs.items()}
    sorted_weighted_averages = sorted(weighted_averages.items(), key=lambda item: item[1])
    if len(sorted_weighted_averages) > 4:
        top_4_keys = [item[0] for item in sorted_weighted_averages[:4]]
    else:
        top_4_keys = [item[0] for item in sorted_weighted_averages]
    
    # Retrieve paths and titles
    out_paths = [img_paths_to_int_id[key] for key in top_4_keys]
    out_titles = [intid_to_item_name[key] for key in top_4_keys]

    print("Retrieved paths:", out_paths)
    print("Retrieved titles:", out_titles)

if __name__ == "__main__":
    main()
