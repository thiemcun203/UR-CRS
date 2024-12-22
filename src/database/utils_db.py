import os
import sys
import logging

import weaviate
from weaviate.client import Client
from weaviate.collections import Collection

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)

# Set up logging
if not os.path.exists("logs"): os.makedirs("logs")
logging.basicConfig(level=logging.INFO, filename="logs/database.log", format="%(asctime)s - %(levelname)s - %(message)s")

def connect_db_local() -> Client:
    """
    Connect to a local Weaviate instance.
    """
    try:
        client = weaviate.connect_to_local(
            host="localhost",  # Use a string to specify the host
            port=8080,
            grpc_port=50051,
        )

        logging.info(f"[utils_db.py] Connect to local Weaviate instance. Result: {client.is_ready()}")
        logging.info(f"[utils_db.py] List of collections: \n{client.collections.list_all()}")
        return client

    except Exception as e:
        logging.error(f"[utils_db.py] Error connecting to local Weaviate instance: {e}")


def connect_db_remote() -> Client:
    """
    Connect to a remote Weaviate instance.
    """
    try:
        http_host = "nguyenbathiem.tail7bf280.ts.net"
        grpc_host = "nguyenbathiem.tail7bf280.ts.net"

        client = weaviate.connect_to_custom(
            http_host=http_host,        # Hostname for the HTTP API connection
            http_port=443,              # Default is 80, WCD uses 443
            http_secure=True,           # Whether to use https (secure) for the HTTP API connection
            grpc_host=grpc_host,        # Hostname for the gRPC API connection
            grpc_port=10000,            # Default is 50051, WCD uses 443
            grpc_secure=True,           # Whether to use a secure channel for the gRPC API connection
        )

        logging.info(f"[utils_db.py] Connect to remote Weaviate instance. Result: {client.is_ready()}")
        logging.info(f"[utils_db.py] List of collections: \n{client.collections.list_all().keys()}")
        return client
    
    except Exception as e:
        logging.error(f"[utils_db.py] Error connecting to remote Weaviate instance: {e}")


def get_movie_collection(client: Client, collection_name: str) -> Collection:
    """
    Get a collection from the Weaviate instance.
    """
    try:
        collection = client.collections.get(name=collection_name)
        logging.info(f"[utils_db.py] Collection {collection_name} retrieved. Found {collection.aggregate.over_all(total_count=True).total_count} objects.")
        return collection
    
    except Exception as e:
        logging.error(f"[utils_db.py] Error getting collection: {e}")

if __name__ == "__main__":
    # client: Client = connect_db_remote()
    client: Client = connect_db_local()
    print(client.collections.list_all().keys())