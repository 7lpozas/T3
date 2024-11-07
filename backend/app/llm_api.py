import aiohttp
import asyncio
import time
import logging
from .retry_utils import retry_async
import json
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAPI:
    def __init__(self):
        self.base_url = "tormenta.ing.puc.cl/api"  
        self.model = "integra-LLM"
        self.rate_limit = 10  
        self.last_request_time = 0

    async def _wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < 1 / self.rate_limit:
            await asyncio.sleep(1 / self.rate_limit - time_since_last_request)
        self.last_request_time = time.time()

    async def generate_completion(self, prompt):
        await self._wait_for_rate_limit()
        url = f"http://{self.base_url}/generate"  
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": 6,
                "num_ctx": 2048,
                "repeat_last_n": 10,
                "top_k": 18
            }
        }

        return await retry_async(lambda: self._make_api_call(url, payload))

    async def _make_api_call(self, url, payload):
        async with aiohttp.ClientSession() as session:
            try:
                #logger.info(f"Haciendo llamada a la API: {url}")
                async with session.post(url, json=payload, timeout=300) as response:  
                    #logger.info(f"Respuesta recibida. Status: {response.status}")
                    if response.status == 200:
                        full_response = ""
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    if 'response' in data:
                                        full_response += data['response']
                                    if data.get('done', False):
                                        break
                                except json.JSONDecodeError:
                                    logger.error(f"Error decodificando JSON: {line}")
                        #logger.info(f"Respuesta completa: {full_response}")
                        return full_response
                    else:
                        logger.error(f"Error en la API del LLM: Status {response.status}, Body: {await response.text()}")
                        return "Lo siento, no puedo generar una respuesta en este momento."
            except asyncio.TimeoutError:
                logger.error("Timeout al conectar con la API del LLM")
                return "Lo siento, la respuesta tardó demasiado. Por favor, intenta con una pregunta más corta o específica."
            except aiohttp.ClientError as e:
                logger.error(f"Error de conexión con la API del LLM: {str(e)}")
                return "Lo siento, hay un problema de conexión. Por favor, intenta más tarde."
            except Exception as e:
                logger.error(f"Error inesperado: {str(e)}", exc_info=True)
                return "Lo siento, ocurrió un error inesperado. Por favor, intenta más tarde."
                
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_chat_completion(self, messages):
        await self._wait_for_rate_limit()
        url = f"http://{self.base_url}/chat"  
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": 6,
                "num_ctx": 2048,
                "repeat_last_n": 10,
                "top_k": 18
            }
        }

        return await retry_async(lambda: self._make_api_call(url, payload))

llm_api = LLMAPI()