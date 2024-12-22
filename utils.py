import os
import sys
import yaml
import logging
from dotenv import load_dotenv

root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root)

from langchain_together.chat_models import ChatTogether
from langchain_together.embeddings import TogetherEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from openai import OpenAI

# Set up logging
if not os.path.exists("logs"): os.makedirs("logs")
logging.basicConfig(level=logging.INFO, filename="logs/app.log", format="%(asctime)s - %(levelname)s - %(message)s")

def get_config():
    """
    Load Configuration
    """
    try:
        config_path = os.path.join(root, "config.yaml")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        logging.info(f"[utils.py] Configuration loaded from {config_path}")
        return config
    
    except Exception as e:
        logging.error(f"[utils.py] Error loading configuration: {e}")


def load_chat_model() -> ChatTogether:
    """
    Load Chat Model
    """
    try:
        load_dotenv()
        config = get_config()

        TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        CHAT_MODEL       = config["model"]["chat_model"]        

        logging.info(f"[utils.py] Together API Key: {'*' * len(TOGETHER_API_KEY)}")
        logging.info(f"[utils.py] Chat Model: {CHAT_MODEL}")

        llm = ChatTogether(
            api_key=TOGETHER_API_KEY,
            model=CHAT_MODEL,
        )

        logging.info(f"[utils.py] Chat model loaded: {CHAT_MODEL}")
        
        return llm
    
    except Exception as e:
        logging.error(f"[utils.py] Error loading chat model: {e}")


def load_embed_model() -> TogetherEmbeddings:
    """
    Load Embedding Model
    """
    try:
        load_dotenv()
        config = get_config()

        TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        EMBED_MODEL      = config["model"]["embed_model"]

        logging.info(f"[utils.py] Together API Key: {'*' * len(TOGETHER_API_KEY)}")
        logging.info(f"[utils.py] Embed Model: {EMBED_MODEL}")

        embed_model = TogetherEmbeddings(
            api_key=TOGETHER_API_KEY,
            model=EMBED_MODEL
        )

        logging.info(f"[utils.py] Embedding model loaded: {EMBED_MODEL}")

        return embed_model
    
    except Exception as e:
        logging.error(f"[utils.py] Error loading embedding model: {e}")


def load_openai_model() -> ChatOpenAI:
    """
    Load OpenAI Chat Model
    """
    try:
        load_dotenv()
        config = get_config()

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_MODEL   = config["openai"]["model_name"]

        logging.info(f"[utils.py] OpenAI API Key: {'*' * len(OPENAI_API_KEY)}")
        logging.info(f"[utils.py] OpenAI Model: {OPENAI_MODEL}")

        openai_model = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL
        )

        logging.info(f"[utils.py] OpenAI model loaded: {OPENAI_MODEL}")

        return openai_model
    
    except Exception as e:
        logging.error(f"[utils.py] Error loading OpenAI model: {e}")


def load_openai_client() -> OpenAI:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key = OPENAI_API_KEY)
    return openai_client


def test_load_embed_model():
    embed_model = load_embed_model()
    query = "Solider amusement park"
    vector = embed_model.embed_query(text=query)
    print(f"[+] Vector: {vector}")

def test_load_chat_model():
    chat_model = load_chat_model()
    query = "What is the capital of France?"
    response = chat_model.invoke(input=query)
    print(f"[+] Response: {response}")

def test_load_openai_model():
    openai_model = load_openai_model()
    query = "What is the capital of France?"
    response = openai_model.invoke(input=query)
    print(f"[+] Response: {response}")

if __name__ == "__main__":
    test_load_openai_model()