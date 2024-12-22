import os
import sys
from typing import List, Dict
from weaviate.collections import Collection
from langchain_core.prompts import PromptTemplate
import logging
import json
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import time
if not os.path.exists("logs"): os.makedirs("logs")
logging.basicConfig(filename="logs/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)

from utils import load_openai_client, load_chat_model
from src.database.utils_db import get_movie_collection


class SearchMovieAPI:
    def __init__(self, collection: Collection) -> None:
        self.collection = collection
    
    def search(self, query_embedding, k=10, search_type="vector", query=""):

        if search_type == "vector":
            # return []
            response = self.collection.query.near_vector(
                near_vector=query_embedding.tolist(), # your query vector goes here
                limit=k,
                # return_metadata=MetadataQuery(distance=True)
            )
            return [obj.properties for obj in response.objects]
            
        elif search_type == "hybrid":
            response = self.collection.query.hybrid(
                query=query,
                vector=query_embedding.tolist(),
                alpha=0.5,
                limit=k,
            )
            return [obj.properties for obj in response.objects]




def infer_chat(chat, model="gpt-4o-mini"):
    openai_client = load_openai_client()
    for i in range(5):
        try:
            chat_response = openai_client.chat.completions.create(
                model=model,
                messages=chat,
                stream=False,
                temperature=0,
            )

            return chat_response.choices[0].message.content
        
        except Exception as e:
            print(e)
            continue


def mean_pooling(model_output, attention_mask):
    # return model_output[0][:, 0]
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#! TODO: Add Search Tool
class MovieChatbotV2:
    def __init__(self, movie_collection: Collection=None) -> None:
        self.chat_history: List[Dict[str, str]] = []
        self.search_tool = SearchMovieAPI(collection=movie_collection)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(self.device)


    def __call__(self, message: str):
        self._update_chat_history(message, "user")

        chat_history_list = []
        for chat_message in self.chat_history:
            role = chat_message["role"]
            content = chat_message["content"]
            
            if role=="user":
                chat_history_list.append(f"B: {content}")
            
            if role=="assistant":
                chat_history_list.append(f"A: {content}")
        t1 = time.time()
        chat_history_text = "\n".join(chat_history_list)
        print(f"Chat history text:\n{chat_history_text}")
        self.reprompt = f"""Given the following text or conversation, identify the user's query as clearly and concisely as possible based on information in coversation. Include the key action, goal, or need expressed by the user in a single sentence. Avoid excessive detail and prioritize clarity. Read carefully conversation to provide user query
Conversation:
{chat_history_text}
"""
        chat = [
            {
                "role": "user",
                "content": self.reprompt
            },
        ]

        new_query = infer_chat(chat=chat)
        print(f"New query: {new_query}")    
        new_query = new_query.split("User query:")[1].strip() if "User query:" in new_query else new_query
        print(f"New query: {new_query}")
        # embeddings = self.model.encode(self.chat_history[-1]["content"])
        embeddings = self.model.encode(new_query)
        print(f"Final user: {self.chat_history[-1]['content']}")

        # Search for movies
        movies: List[Dict[str, str]] = self.search_tool.search(query_embedding=embeddings, k=10, search_type="vector")
        # print(f"Movies found:\n{movies[0]}")
        print(f"Movies found: {[movie['title'] + movie['movie_id'] for movie in movies]}")
        movies_dict = {
            movie["movie_id"]: f"Movie title: {movie['title']}, Genre: {movie['genre']}, Year: {movie['year']}, Plot: {movie['plot']}, Actors: {movie['actors']}, Director: {movie['director']}, Writer: {movie['writer']}" 
            for movie in movies
        } if movies else {}

        prompt = """You are a movie recommendation system. Based on the user’s conversation and available movie data, generate a response and reorder all movie IDs in movies_id_list to present the most suitable movies.
If the user’s intent is unrelated to movies or there’s insufficient information:
•Respond naturally by answering or asking for more details.
•Set movies_id_list to null.
Movies_id_list:
•Ensure movies_id_list contains all movie IDs reordered from highest suitable to lowest, not just related ones.
Response:
•Make it concise, engaging, and satisfying for the user.
•Only include movies from the provided list; do not create new entries and repeat the movie mentioned in conversation.
Output Format:
{
    "response": "<your response here>",
    "movies_id_list": ["movie_id1", "movie_id2", "..."] or null
}""" + f"""Here is movies id list (id:information):
{json.dumps(movies_dict)}"""

        conv_chat = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "user",
                "content": chat_history_text
            }
        ]

        try:
            response_dict = infer_chat(chat=conv_chat)
            logging.info(f"Response from OpenAI:\n{response_dict}")

            response_dict = json.loads(response_dict)
            response_text = response_dict.get("response", "")
            logging.info(f"Response text: {response_text}")

            movies_id_list = response_dict.get("movies_id_list", None)
            logging.info(f"Movies ID list: {movies_id_list}")
            
            self._update_chat_history(response_text, "assistant")


            # Create movies lookup dictionary
            movie_lookup = {movie["movie_id"]: movie for movie in movies}
            ordered_movies = [
                movie_lookup[movie_id] for movie_id in movies_id_list
            ] if movies_id_list is not None else []
            
            response = {
                "response": response_text,
                "movies": [
                    {
                        "title": movie["title"],
                        "genre": movie["genre"],
                        "year": movie["year"],
                        "thumbnail": f"/static/assets/thumbnails/{movie['movie_id']}.jpg",
                        "plot": movie["plot"],
                        "actors": movie["actors"],
                        "director": movie["director"],
                        "writer": movie["writer"],
                    }
                    for movie in ordered_movies[:5]
                ] if ordered_movies else []
            }
            # print(f"Response: {response}")
            t2 = time.time()
            print(f"Time to process: {t2-t1}")
            return response

        except Exception as e:
            logging.error(f"Error in chatbot: {e}")

            error_message = "I'm sorry, I couldn't find any movies for you."

            self._update_chat_history(error_message, "assistant")
            t2 = time.time()
            print(f"Time to process: {t2-t1}")
            return {
                "response": error_message,
                "movies": []
            }


    def start_new_session(self):
        """
        Start a new session
        """
        self.chat_history = [
            {
                "role": "assistant",
                "content": "Hi! How can I help you?"
            }
        ]
        logging.info(f"New session started")
        logging.info(f"Chat history: {self.chat_history}")


    def _update_chat_history(self, message: str, role: str):
        """
        Update the chat history
        Args:
            message (str): The message to add to the chat history
            role (str): The role of the message
        """
        self.chat_history.append({"role": role, "content": message})
        logging.info(f"Chat history updated: {self.chat_history}")