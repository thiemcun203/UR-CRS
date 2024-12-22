# app.py
import os
import sys

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Set up logging for app
import logging
from datetime import datetime

if not os.path.exists("logs"): 
    os.makedirs("logs")

# Format timestamp for valid filename
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO, 
    filename=f"logs/app-{log_timestamp}.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

from src.database.utils_db import get_movie_collection, connect_db_remote, connect_db_local
from utils import get_config
from src.components.chatbot import MovieChatbotV2


# ---------------------------------------------
# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 template rendering
templates = Jinja2Templates(directory="template")
app.mount("/static", StaticFiles(directory="static"), name="static")



# Connect to Movie Database & Initialize Chatbot
config = get_config()
COLLECTION_NAME = config["database"]["collection_name"]

client = connect_db_local() # client = connect_db_remote()
movie_collection = get_movie_collection(client=client, collection_name=COLLECTION_NAME)

# chatbot = MovieChatbot(movie_collection=movie_collection)
chatbot = MovieChatbotV2(movie_collection=movie_collection)


# ---------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the chat interface
    """
    chatbot.start_new_session() # Reset session on page load
    
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(query: str):
    """
    Handle chat requests and generate responses.
    """
    response_data = chatbot(query)

    return JSONResponse(content=response_data)


# ---------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8097)