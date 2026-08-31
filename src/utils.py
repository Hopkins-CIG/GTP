from pathlib import Path

def get_project_root() -> Path:
    """
    Get the root directory of the project.
    """
    return Path(__file__).parent.parent