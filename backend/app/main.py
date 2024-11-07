from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.rag_system import rag_system
from app.llm_api import llm_api
from app.script_processor import run_script_processing
from pydantic import BaseModel
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://*.onrender.com", "https://*s.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MovieCreate(BaseModel):
    title: str
    url: str

class QueryCreate(BaseModel):
    question: str

movies_to_process = [
    {"title": "Ted", "url": "https://imsdb.com/scripts/Ted.html"},
    {"title": "Rush Hour", "url": "https://imsdb.com/scripts/Rush-Hour.html"},
    {"title": "Rush Hour 2", "url": "https://imsdb.com/scripts/Rush-Hour-2.html"},
    {"title": "Passengers", "url": "https://imsdb.com/scripts/Passengers.html"},
    {"title": "Oblivion", "url": "https://imsdb.com/scripts/Oblivion.html"},
    {"title": "Limitless", "url": "https://imsdb.com/scripts/Limitless.html"},
    {"title": "American Pie", "url": "https://imsdb.com/scripts/American-Pie.html"},
    {"title": "Godfather Part II", "url": "https://imsdb.com/scripts/Godfather-Part-II.html"},
    {"title": "Dune", "url": "https://imsdb.com/scripts/Dune.html"},
    {"title": "Dune Part One", "url": "https://imsdb.com/scripts/Dune-Part-One.html"},
]

@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando la aplicación...")
    if os.path.exists(rag_system.index_path) and os.path.exists(rag_system.embeddings_path):
        rag_system.load_index_and_embeddings()
    else:
        try:
            processed_dir, processed_movies = await run_script_processing(movies_to_process)
            if processed_movies:
                try:
                    await rag_system.process_scripts(processed_dir)
                except Exception as e:
                    logger.error(f"Error processing scripts with RAG system: {str(e)}")
            else:
                logger.warning("No se procesaron películas debido a errores.")
        except Exception as e:
            logger.error(f"Error durante el inicio de la aplicación: {str(e)}")

@app.post("/query")
async def process_query(query: QueryCreate):
    try:
        context = await rag_system.get_relevant_context(query.question)
        prompt = f"Basándote en el siguiente contexto sobre películas, responde a la pregunta: '{query.question}'\n\nContexto: {context}\n\nRespuesta:"
        response = await llm_api.generate_completion(prompt)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movies")
def get_movies():
    return movies_to_process

@app.get("/check_movies")
def check_movies():
    return [{"title": movie['title'], "url": movie['url']} for movie in movies_to_process]

@app.get("/faiss_status")
async def check_faiss_status():
    if rag_system.vectorstore is not None:
        return {"status": "FAISS index is loaded and ready", "index_size": len(rag_system.vectorstore.index_to_docstore_id)}
    else:
        return {"status": "FAISS index is not loaded"}

@app.get("/test_query")
async def test_faiss_query(query: str):
    if rag_system.vectorstore is None:
        raise HTTPException(status_code=500, detail="FAISS index is not loaded")
    
    try:
        context = await rag_system.get_relevant_context(query, k=3)
        return {"query": query, "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")

@app.get("/check_files")
def check_files():
    index_path = "faiss_index"
    embeddings_path = "embeddings.pkl"
    return {
        "faiss_index_exists": os.path.exists(index_path),
        "embeddings_file_exists": os.path.exists(embeddings_path)
    }

@app.post("/force_index_creation")
async def force_index_creation():
    try:
        processed_dir = "processed_scripts"  
        await rag_system.process_scripts(processed_dir)
        return {"status": "Index creation process completed"}
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating index: {str(e)}")

#if __name__ == "__main__":
 #   import uvicorn
  #  uvicorn.run(app, host="0.0.0.0", port=8000)