from pathlib import Path
from typing import Union, List, Literal

def read_file_name(file: Union[str, Path], extension:Literal['csv', 'dat'] = 'csv') -> List[str]:
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {file}")
    if path.is_file():
        # Return the file name itself
        if path.name.lower().lstrip('.') != extension:
            return []
        else:
            return [path.name]
    elif path.is_dir():
        # Return all files in directory
        all_files = sorted([f.name for f in path.iterdir() if f.is_file()])
        return [f for f in all_files if f.lower().endswith(f'.{extension}')]
    else:
        raise ValueError(f"Path is neither a file nor directory: {file}")
