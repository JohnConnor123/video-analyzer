import requests
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

class WhisperClient:
    """HTTP-клиент для обращения к Whisper API"""
    
    def __init__(self, api_url: str, timeout: int = 300):
        """
        Инициализация клиента для Whisper API
        
        Args:
            api_url: URL API Whisper сервиса (например, http://localhost:16000)
            timeout: Таймаут запроса в секундах
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.transcribe_url = f"{self.api_url}/v1/audio/transcriptions"
        
        # Проверка доступности сервиса при инициализации
        try:
            health_response = requests.get(
                f"{self.api_url}/health", 
                timeout=10
            )
            if health_response.status_code == 200:
                health_data = health_response.json()
                logger.info(f"Whisper API доступен: {health_data}")
            else:
                logger.warning(f"Whisper API вернул статус {health_response.status_code}")
        except Exception as e:
            logger.error(f"Не удалось проверить доступность Whisper API: {e}")
            raise ConnectionError(f"Whisper API недоступен по адресу {api_url}")
    
    def transcribe(self, audio_path: Path, 
                  language: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Транскрипция аудио файла через HTTP API
        
        Args:
            audio_path: Путь к аудио файлу
            language: Код языка (опционально)
            
        Returns:
            Результат транскрипции или None в случае ошибки
        """
        try:
            # Подготовка файла для отправки
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'file': (audio_path.name, audio_file, 'audio/wav')
                }
                
                # Подготовка параметров формы
                data = {
                    'model': 'whisper-1',  # Совместимость с OpenAI API
                    'response_format': 'json'
                }
                
                if language:
                    data['language'] = language
                
                logger.debug(f"Отправка запроса на транскрипцию: {audio_path}")
                
                # Отправка запроса
                response = requests.post(
                    self.transcribe_url,
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.debug("Транскрипция завершена успешно")
                return result
                
        except requests.exceptions.Timeout:
            logger.error(f"Таймаут при транскрипции файла {audio_path}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка HTTP при транскрипции: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при транскрипции: {e}")
            return None
    
    def check_health(self) -> bool:
        """
        Проверка доступности Whisper API
        
        Returns:
            True если API доступен, False иначе
        """
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False 