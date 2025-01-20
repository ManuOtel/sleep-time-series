"""
This module handles API key generation and management for the Sleep Stage Classification API.
(No, this is not the kind of keygen one may used to download from shady websites to crack MS Office or NFS Most Wanted! 😂😂)

The main purpose is to generate secure API keys and store them in the environment file.
The module provides functionality to:
    1. Generate cryptographically secure random API keys
    2. Create or update the .env file with the generated key
    3. Validate the .env file structure and permissions
    4. Provide command line interface for key generation

The generated API keys are used by the FastAPI application for request authentication.
"""

import secrets
import string
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate_api_key(length: int = 32) -> str:
    """Generate a secure random API key.

    Creates a cryptographically secure random string using only alphanumeric characters
    to avoid issues with special characters in environment variables.

    Args:
        length: Length of the API key to generate. Defaults to 32 characters.
            Minimum recommended length is 32.

    Returns:
        A randomly generated API key string containing only letters and numbers

    Raises:
        ValueError: If length is less than 32 characters
    """
    if length < 32:
        raise ValueError("API key length must be at least 32 characters")

    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def update_env_file(api_key: str, env_path: Optional[Path] = None) -> None:
    """Update or create .env file with the provided API key.

    Writes the API key to the .env file, creating the file if it doesn't exist
    and preserving any other environment variables if it does.

    Args:
        api_key: The API key string to write to the file
        env_path: Optional path to .env file. Defaults to .env in current directory.

    Returns:
        None

    Raises:
        PermissionError: If unable to write to the .env file
        OSError: If there are other IO-related errors
    """
    if env_path is None:
        env_path = Path('.env')

    logger.info(f"Updating API key in {env_path}")

    try:
        # Read existing env file content
        env_content = {}
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        env_content[key] = value

        # Update API key
        env_content['MANU_API_KEY'] = api_key

        # Write back to file
        with open(env_path, 'w') as f:
            for key, value in env_content.items():
                f.write(f"{key}='{value}'\n")

        # Set restrictive permissions
        env_path.chmod(0o600)

        logger.info("Successfully updated API key")

    except Exception as e:
        logger.error(f"Failed to update API key: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate API key and update .env file')
    parser.add_argument('--length', type=int, default=32,
                        help='Length of API key to generate (minimum 32)')
    parser.add_argument('--env-file', type=Path, default=None,
                        help='Path to .env file (default: .env in current directory)')

    args = parser.parse_args()

    try:
        api_key = generate_api_key(args.length)
        update_env_file(api_key, args.env_file)
        print("API key generated and stored successfully")
    except Exception as e:
        logger.error(f"Failed to generate/store API key: {str(e)}")
        raise
