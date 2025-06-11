import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import subprocess
from pydub import AudioSegment
from .clients.whisper_client import WhisperClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AudioTranscript:
    text: str
    segments: List[Dict[str, Any]]
    language: str

class AudioProcessor:
    def __init__(self, 
                 language: str | None = None,
                 whisper_api_url: str = "http://localhost:16000",
                 timeout: int = 300):
        """
        Инициализация аудиопроцессора с HTTP-клиентом для Whisper API
        
        Args:
            language: Код языка для транскрипции
            whisper_api_url: URL Whisper API
            timeout: Таймаут запросов в секундах
        """
        try:
            self.language = language if language else None

            # Инициализация HTTP-клиента для Whisper
            self.whisper_client = WhisperClient(
                api_url=whisper_api_url,
                timeout=timeout
            )
            
            logger.info(f"Инициализация AudioProcessor: Whisper API URL: {whisper_api_url}, Язык: {self.language if self.language else 'автоопределение'}")
            
            # Проверка наличия ffmpeg
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                self.has_ffmpeg = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.has_ffmpeg = False
                logger.warning("FFmpeg не найден. Установите ffmpeg для лучшего извлечения аудио.")
                
        except Exception as e:
            logger.error(f"Ошибка инициализации AudioProcessor: {e}")
            raise

    def extract_audio(self, video_path: Path, output_dir: Path) -> Optional[Path]:
        """Extract audio from video file and convert to format suitable for Whisper.
        Returns None if video has no audio streams."""
        audio_path = output_dir / "audio.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract audio using ffmpeg
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit little-endian
                "-ar", "16000",  # 16kHz sampling rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                str(audio_path)
            ], check=True, capture_output=True)
            
            logger.debug("Successfully extracted audio using ffmpeg")
            return audio_path
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode()
            logger.error(f"FFmpeg error: {error_output}")
            
            # Check if error indicates no audio streams
            if "Output file does not contain any stream" in error_output:
                logger.debug("No audio streams found in video - skipping audio extraction")
                return None
                
            # If error is not about missing audio, try pydub as fallback
            logger.info("Falling back to pydub for audio extraction...")
            try:
                video = AudioSegment.from_file(str(video_path))
                audio = video.set_channels(1).set_frame_rate(16000)
                audio.export(str(audio_path), format="wav")
                logger.debug("Successfully extracted audio using pydub")
                return audio_path
            except Exception as e2:
                logger.error(f"Error extracting audio using pydub: {e2}")
                # If both methods fail, raise error
                raise RuntimeError(
                    "Failed to extract audio. Please install ffmpeg using:\n"
                    "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
                    "MacOS: brew install ffmpeg\n"
                    "Windows: choco install ffmpeg"
                )

    def transcribe(self, audio_path: Path) -> Optional[AudioTranscript]:
        """Транскрипция аудио файла через Whisper API"""
        try:
            # Отправка запроса к Whisper API
            result = self.whisper_client.transcribe(
                audio_path=audio_path,
                language=self.language
            )
            
            if not result:
                logger.warning("Не удалось получить результат транскрипции")
                return None
            
            # Обработка ответа от API
            text = result.get('text', '')
            if not text or not text.strip():
                logger.warning("Текст не обнаружен в аудио")
                return None
            
            # Обработка сегментов, если они есть
            segments = result.get('segments', [])
            if not segments:
                # Создаем базовый сегмент, если сегменты отсутствуют
                segments = [{
                    "text": text,
                    "start": 0.0,
                    "end": 0.0,
                    "words": []
                }]
            
            # Преобразование к ожидаемому формату
            segment_data = []
            for segment in segments:
                segment_info = {
                    "text": segment.get("text", ""),
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "words": []
                }
                
                # Обработка слов, если они есть
                if "words" in segment and segment["words"]:
                    for word in segment["words"]:
                        word_info = {
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0),
                            "probability": word.get("probability", 1.0)
                        }
                        segment_info["words"].append(word_info)
                
                segment_data.append(segment_info)
            
            # Определение языка
            detected_language = result.get('language', self.language or 'unknown')
            
            return AudioTranscript(
                text=text,
                segments=segment_data,
                language=detected_language
            )
            
        except Exception as e:
            logger.error(f"Ошибка транскрипции аудио: {e}")
            logger.exception(e)
            return None
