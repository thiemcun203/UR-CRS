import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from collections import defaultdict

import os, weaviate
import weaviate
from weaviate.classes.init import Auth
from weaviate.util import generate_uuid5
from weaviate.classes.config import Property, DataType, Tokenization, Configure, VectorDistances
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery

import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)

from src.database.utils_db import connect_db_remote, connect_db_local
from utils import get_config


def mean_pooling(model_output, attention_mask):
    # return model_output[0][:, 0]
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class MovieEmbeddingDatabase:
    def __init__(self, device='cuda'):
        self.device = device
        
        # self.client = weaviate.connect_to_local()
        # self.client = connect_db_remote()
        self.client = connect_db_local()
        config = get_config()

        self.collection_name = config["database"]["collection_name"]
        self.init_collection(self.collection_name)
        
    def init_collection(self, collection_name):
        self.client.connect()
        self.client.collections.delete(collection_name)
        self.client.collections.create(
        collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(
                name="content",
                data_type=DataType.TEXT,
                tokenization=Tokenization.LOWERCASE,
                index_filterable=True,
                index_searchable=True,
            ),
            Property(
                name="obj_id",
                data_type=DataType.INT,
                # tokenization=Tokenization.LOWERCASE,
                index_filterable=True,
                # index_searchable=True,
            ),
                Property(
                    name="movie_id",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.LOWERCASE,
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.LOWERCASE,
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="year",
                    data_type=DataType.TEXT,
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="genre",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.LOWERCASE,
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="director",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.LOWERCASE,
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="actors",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.LOWERCASE,
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="writer",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.LOWERCASE,
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="plot",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.LOWERCASE,
                    index_filterable=True,
                    index_searchable=False,
                ),
        ],
        inverted_index_config=wvc.config.Configure.inverted_index(
                index_null_state=True,
                index_property_length=True,
                index_timestamps=True,
                stopwords_removals=[],
                bm25_k1=1.2,
                bm25_b=0.75,
            ),
        vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
        )
        
    def compute_embedding(self, model, tokenizer, texts):
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            
            return_tensors='pt',
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            output = model(**encoded)
            embeddings = mean_pooling(output, encoded['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def update_embeddings(self, model, tokenizer, movie_db):
        # Reset collection
        self.init_collection(self.collection_name)
        self.collection = self.client.collections.get(self.collection_name)
        # Process movies in batches
        batch_size = 256
        movie_ids = list(movie_db.keys())
        movie_texts = [movie_db[mid] for mid in movie_ids]
        
        for i in tqdm(range(0, len(movie_ids), batch_size), desc="Updating movie embeddings"):
            batch_ids = movie_ids[i:i + batch_size]
            batch_texts = movie_texts[i:i + batch_size]
            batch_embeddings = self.compute_embedding(model, tokenizer, batch_texts)
            
            data_objects = list()
            for k, content in enumerate(batch_texts):
                # print(type(batch_ids[k]), type(content))

                movie_id = str(batch_ids[k])
                movie_info = mapping_movie_id(movie_id)
                print(movie_info.get("title"))
                data_object = wvc.data.DataObject(
                    properties={
                            "content": content,
                            "obj_id": int(batch_ids[k]),
                            "movie_id": movie_id,
                            "title": movie_info.get("title"),
                            "year": movie_info.get("year"),
                            "genre": movie_info.get("genre"),
                            "director": movie_info.get("director"),
                            "actors": movie_info.get("actors"),
                            "writer": movie_info.get("writer"),
                            "plot": movie_info.get("plot")
                        },

                    uuid=generate_uuid5(content+str(batch_ids[k])),
                    vector=batch_embeddings[k].tolist(),
                )
                if batch_embeddings[k] is None:
                    print("None embedding", k)
                data_objects.append(data_object)
                
            
            self.collection.data.insert_many(data_objects)
            # print(f"Inserted {len(data_objects)} data objects")
        


    def query_id(self, query_embedding, k=10, search_type = "vector", query = ""):
        collection = self.client.collections.get(self.collection_name)
        result = []
    
        if search_type == "vector":
            response = collection.query.near_vector(
                near_vector=query_embedding.tolist(), # your query vector goes here
                limit=k,
                # return_metadata=MetadataQuery(distance=True)
            )
            return response
            
        elif search_type == "hybrid":
            response = collection.query.hybrid(
                query=query,
                vector=query_embedding.tolist(),
                alpha=0.5,
                limit=k,
            )
            return response

    
import os
import pandas as pd
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.path.join(dir_path, "data", "full_redial", "concat_movie_db")

redial_path_csv = os.path.join(dir_path, "data", "redial", "redial_moviedb.csv")
redial_df = pd.read_csv(redial_path_csv)

lookup_redial_movie = {
    str(row["movie_id"]): {
        "movie_id": str(row["movie_id"]),
        "title": row["movie_title"],
        "year": row["year"],
        "genre": row["genre"],
        "director": row["director"],
        "actors": row["actors"],
        "writer": row["writer"],
        "plot": row["plot"]
    } for i, row in redial_df.iterrows()
}

def mapping_movie_id(movie_id: str):
    found = lookup_redial_movie.get(movie_id)
    return found

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
tokenizer.truncation_side = 'left'

model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2', torch_dtype=torch.float32)
model = model.to(device)




movie_db = torch.load(data_path)
# Get first 10 items in dictionary
first_10_items = {k: movie_db[k] for k in list(movie_db)}

# Initialize movie embedding database
movie_embedder = MovieEmbeddingDatabase(device)
movie_embedder.update_embeddings(model, tokenizer, first_10_items)