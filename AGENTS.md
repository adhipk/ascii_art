```
# AGENTS.md - Coding Guidelines and Commands

## Development Commands
- Build: poetry install
- Lint: poetry run flake8
- Test: poetry run pytest
- Format: poetry run black .
- Typecheck: poetry run mypy .

## Code Style Guidelines
- Imports: Use absolute imports
- Formatting: Use black with Python 3.9+ settings
- Naming: Follow PEP 8 conventions
- Types: Use type hints for all function parameters and return values
- Error Handling: Use built-in exception types

## Project Structure
- Source files in root directory
- Tests in tests/ directory
- Assets in assets/ directory
- Configuration in pyproject.toml

## Commit Guidelines
- Use conventional commits format
- Include type hints in commit messages
- Reference issues when applicable
```