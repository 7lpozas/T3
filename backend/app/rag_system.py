import os
import aiohttp
import asyncio
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.base import Embeddings
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pickle
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomEmbeddings(Embeddings):
    def __init__(self):
        self.base_url = "tormenta.ing.puc.cl/api/embed"
        self.model = "nomic-embed-text"
        self.max_retries = 5
        self.retry_delay = 1

    @retry(stop=stop_after_attempt(5), 
           wait=wait_exponential(multiplier=1, min=4, max=60),
           retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)))
    async def _embed_single(self, session, text):
        payload = {
            "model": self.model,
            "input": text
        }
        async with session.post(f"http://{self.base_url}", json=payload, timeout=300) as response:
            if response.status == 200:
                result = await response.json()
                return result['embeddings'][0]
            else:
                raise Exception(f"Error en la API: Status {response.status}, Body: {await response.text()}")

    async def aembed_documents(self, texts):
        async with aiohttp.ClientSession() as session:
            embeddings = []
            for i, text in enumerate(texts):
                success = False
                for attempt in range(self.max_retries):
                    try:
                        embedding = await self._embed_single(session, text)
                        embeddings.append(embedding)
                        logger.info(f"Embedding {i} generado exitosamente. Dimensiones: {len(embedding)}")
                        success = True
                        break
                    except Exception as e:
                        logger.error(f"Error al generar embedding {i} (intento {attempt + 1}): {str(e)}")
                        if attempt == self.max_retries - 1:
                            logger.error(f"No se pudo generar el embedding {i} después de {self.max_retries} intentos.")
                            return None  
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))  
                if not success:
                    return None  
            return embeddings

    async def aembed_query(self, text):
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    return await self._embed_single(session, text)
                except Exception as e:
                    logger.error(f"Error al generar embedding para la consulta (intento {attempt + 1}): {str(e)}")
                    if attempt == self.max_retries - 1:
                        logger.error(f"No se pudo generar el embedding para la consulta después de {self.max_retries} intentos.")
                        return None
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  
            return None

    def embed_documents(self, texts):
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text):
        return asyncio.run(self.aembed_query(text))

class RAGSystem:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.embeddings = CustomEmbeddings()
        self.vectorstore = None
        self.index_path = "faiss_index"
        self.embeddings_path = "embeddings.pkl"

    def save_index_and_embeddings(self):
        if self.vectorstore:
            self.vectorstore.save_local(self.index_path)
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info("Índice FAISS y embeddings guardados exitosamente.")

    def load_index_and_embeddings(self):
        if os.path.exists(self.index_path) and os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            self.vectorstore = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Índice FAISS cargado exitosamente. Tamaño del índice: {len(self.vectorstore.index_to_docstore_id)}")
        else:
            logger.warning("No se encontraron archivos de índice FAISS o embeddings.")
    
    async def get_relevant_context(self, query, k=5):
        start_time = time.time()
        logger.info(f"Iniciando proceso de obtención de contexto para la consulta: {query}")
        
        if not self.vectorstore:
            logger.error("Vectorstore no inicializado. Llama a process_scripts primero.")
            raise ValueError("Vectorstore no inicializado. Llama a process_scripts primero.")
        
        logger.info("Generando embedding para la consulta...")
        query_embedding_start = time.time()
        query_embedding = await self.embeddings.aembed_query(query)
        query_embedding_time = time.time() - query_embedding_start
        logger.info(f"Tiempo de generación del embedding de la consulta: {query_embedding_time:.2f} segundos")
        
        if query_embedding is None:
            logger.error("No se pudo generar el embedding para la consulta")
            return "Lo siento, no puedo procesar tu consulta en este momento."
        
        logger.info(f"Embedding generado para la consulta. Dimensiones: {len(query_embedding)}")
        
        if len(query_embedding) != 768:
            logger.error(f"Vector de consulta tiene dimensión incorrecta: {len(query_embedding)}")
            return "Error en el procesamiento de la consulta."
        
        logger.info(f"Realizando búsqueda de similitud con k={k}...")
        search_start = time.time()
        results = self.vectorstore.similarity_search_by_vector(query_embedding, k=k)
        search_time = time.time() - search_start
        logger.info(f"Tiempo de búsqueda de similitud: {search_time:.2f} segundos")
        
        logger.info(f"Se encontraron {len(results)} resultados relevantes")
        
        context = "\n".join([doc.page_content for doc in results])
        logger.info(f"Contexto generado (primeros 100 caracteres): {context[:100]}...")
        
        total_time = time.time() - start_time
        logger.info(f"Tiempo total de procesamiento de la consulta: {total_time:.2f} segundos")
        
        return context

    async def process_scripts(self, processed_dir):
        documents = []
        for filename in os.listdir(processed_dir):
            if filename.endswith('.txt'):
                loader = TextLoader(os.path.join(processed_dir, filename))
                documents.extend(loader.load())
        
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"Documentos divididos en {len(texts)} fragmentos")
        
        embedded_documents = await self.embeddings.aembed_documents([doc.page_content for doc in texts])
        
        if embedded_documents is None:
            logger.error("No se pudieron generar todos los embeddings. Abortando el proceso.")
            return
        
        for i, emb in enumerate(embedded_documents):
            if len(emb) != 768:
                logger.error(f"Vector {i} tiene dimensión incorrecta: {len(emb)}, debería ser 768")
                return
        
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip([doc.page_content for doc in texts], embedded_documents)),
            embedding=self.embeddings,
            metadatas=[doc.metadata for doc in texts]
        )
        logger.info(f"Índice FAISS creado exitosamente con {len(embedded_documents)} embeddings")
        self.save_index_and_embeddings()

rag_system = RAGSystem()