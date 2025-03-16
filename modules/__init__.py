# Add import to make modules work as a package and avoid import errors
import os
import pathlib
from dotenv import load_dotenv

# Load environment variables at package initialization
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)
