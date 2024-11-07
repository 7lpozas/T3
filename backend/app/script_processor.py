import os
import re
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import chardet

def clean_script(text):

    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(?i)(FADE IN|CUT TO|CONTINUED|BLACKOUT|DISSOLVE TO|CLOSE UP|WIDE SHOT|EXT\.|INT\.)', '', text)
    text = re.sub(r'$$.*?$$', '', text, flags=re.DOTALL)
    text = re.sub(r'\b[A-Z\s]+\b(?=:)', '', text)
    text = re.sub(r'[^A-Za-z0-9\s.,!?\'-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text

async def download_script(session, url, title):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                encoding = chardet.detect(content)['encoding']
                text = content.decode(encoding or 'utf-8', errors='replace')
                
                soup = BeautifulSoup(text, 'html.parser')
                script_text = soup.find('pre').text if soup.find('pre') else "Script not found"
                
                os.makedirs('scripts', exist_ok=True)
                
                filename = f'scripts/{title}.txt'
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(script_text)
                print(f"Script for {title} downloaded successfully.")
                return filename
            else:
                print(f"Failed to download script for {title}. Status code: {response.status}")
                return None
    except Exception as e:
        print(f"Error downloading script for {title}: {str(e)}")
        return None

def process_script(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            script_text = file.read()
        
        cleaned_text = clean_script(script_text)
        
        processed_dir = 'processed_scripts'
        os.makedirs(processed_dir, exist_ok=True)
        
        processed_filename = os.path.join(processed_dir, os.path.basename(filename))
        with open(processed_filename, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)
        
        print(f"Processed {filename}")
        return processed_filename
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None

async def download_and_process_scripts(movie_list):
    async with aiohttp.ClientSession() as session:
        download_tasks = [download_script(session, movie['url'], movie['title']) for movie in movie_list]
        script_filenames = await asyncio.gather(*download_tasks)
    
    processed_movies = []
    for movie, filename in zip(movie_list, script_filenames):
        if filename:
            processed_filename = process_script(filename)
            if processed_filename:
                movie['script_path'] = processed_filename
                processed_movies.append(movie)
    
    return 'processed_scripts', processed_movies

async def run_script_processing(movie_list):
    return await download_and_process_scripts(movie_list)