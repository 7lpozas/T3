import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def retry_async(func, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Máximo número de intentos alcanzado. Último error: {str(e)}")
                raise
            logger.warning(f"Intento {attempt + 1} fallido. Reintentando en {delay} segundos...")
            await asyncio.sleep(delay)
            delay *= 2 