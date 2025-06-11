from pathlib import Path
import logging
from typing import List, Dict

try:
    # Python 3.9+
    from importlib.resources import files
except ImportError:
    # Python 3.8
    from importlib_resources import files

logger = logging.getLogger(__name__)

class PromptLoader:
    def __init__(self, prompt_dir: str, prompts: List[Dict[str, str]]):
        # Handle user-provided prompt directory
        self.prompt_dir = Path(prompt_dir).expanduser() if prompt_dir else None
        self.prompts = prompts
        self._package_prompt_content = None

    def _find_prompt_file(self, prompt_path: str) -> Path:
        """Find prompt file in package resources, package directory, or user directory."""
        # First try package resources (works for both install modes)
        try:
            package_resource = files('video_analyzer') / 'prompts' / prompt_path
            if package_resource.is_file():
                # For Python 3.9+ we can convert to Path directly
                # For older versions, we need to use a context manager
                try:
                    # Try to get the path directly (works in some cases)
                    return Path(str(package_resource))
                except (AttributeError, TypeError):
                    # Fallback: create temporary file if needed
                    with package_resource.open('r') as f:
                        content = f.read()
                    # Since we need to return a Path, let's save content and create temp path
                    # But actually, let's modify the calling code to handle this better
                    self._package_prompt_content = content
                    return Path("__PACKAGE_RESOURCE__")  # Special marker
        except Exception as e:
            logger.debug(f"Could not find package prompt via importlib.resources: {e}")

        # Try package directory (for development mode)
        pkg_root = Path(__file__).parent
        pkg_path = pkg_root / 'prompts' / prompt_path
        if pkg_path.exists():
            return pkg_path

        # Finally try user-specified directory if provided
        if self.prompt_dir:
            user_path = Path(self.prompt_dir).expanduser()
            # Try absolute path
            if user_path.is_absolute():
                full_path = user_path / prompt_path
                if full_path.exists():
                    return full_path
            else:
                # Try relative to current directory
                cwd_path = Path.cwd() / self.prompt_dir / prompt_path
                if cwd_path.exists():
                    return cwd_path

        raise FileNotFoundError(
            f"Prompt file not found in package resources, package directory, or user directory ({self.prompt_dir})"
        )

    def get_by_index(self, index: int) -> str:
        """Load prompt from file by index.
        
        Args:
            index: Index of the prompt in the prompts list
            
        Returns:
            The prompt text content
            
        Raises:
            IndexError: If index is out of range
            FileNotFoundError: If prompt file doesn't exist
        """
        try:
            if index < 0 or index >= len(self.prompts):
                raise IndexError(f"Prompt index {index} out of range (0-{len(self.prompts)-1})")
            
            prompt = self.prompts[index]
            prompt_path = self._find_prompt_file(prompt["path"])
                
            # Check if this is package resource content
            if str(prompt_path) == "__PACKAGE_RESOURCE__":
                logger.debug(f"Loading prompt '{prompt['name']}' from package resources")
                return self._package_prompt_content.strip()
            else:
            logger.debug(f"Loading prompt '{prompt['name']}' from {prompt_path}")
            with open(prompt_path) as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error loading prompt at index {index}: {e}")
            raise

    def get_by_name(self, name: str) -> str:
        """Load prompt from file by name.
        
        Args:
            name: Name of the prompt to load
            
        Returns:
            The prompt text content
            
        Raises:
            ValueError: If prompt name not found
            FileNotFoundError: If prompt file doesn't exist
        """
        try:
            prompt = next((p for p in self.prompts if p["name"] == name), None)
            if prompt is None:
                raise ValueError(f"Prompt with name '{name}' not found")
            
            prompt_path = self._find_prompt_file(prompt["path"])
                
            # Check if this is package resource content
            if str(prompt_path) == "__PACKAGE_RESOURCE__":
                logger.debug(f"Loading prompt '{name}' from package resources")
                return self._package_prompt_content.strip()
            else:
            logger.debug(f"Loading prompt '{name}' from {prompt_path}")
            with open(prompt_path) as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error loading prompt '{name}': {e}")
            raise
