import os
import pandas as pd
import requests
from tqdm import tqdm


failed_urls = {
    "failed_urls": [],
    "movie_id": []
}

def download_image(image_url: str, movie_id: str, output_dir='thumbnails'):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    try:
        response = requests.get(image_url)
        
        image_name = os.path.join(output_dir, f"{movie_id}.jpg")
        
        with open(image_name, 'wb') as file:
            file.write(response.content)
        
    except Exception as e:
        failed_urls["failed_urls"].append(image_url)
        failed_urls["movie_id"].append(movie_id)


def run_all():
    ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    INPUT_DIR  = os.path.join(ROOT_DIR, "data", "redial", "redial_moviedb.csv")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "thumbnails")

    df = pd.read_csv(INPUT_DIR)

    for i in tqdm(range(len(df)), desc="Downloading thumbnails"):
        movie_id = df.loc[i, "movie_id"]
        image_url = df.loc[i, "poster"]
        download_image(image_url, movie_id, output_dir=OUTPUT_DIR)

    # Export failed URLs as CSV
    failed_urls_df = pd.DataFrame(failed_urls)
    failed_urls_df.to_csv(os.path.join(ROOT_DIR, "data", "failed_urls.csv"), index=False)

if __name__ == "__main__":
    run_all()