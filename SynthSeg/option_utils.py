import os
from typing import Optional


def get_absolute_path(filename: str, reference_file: Optional[str] = None) -> str:
    """
    Function to get the absolute path of a file where reference_file is used to resolve relative paths.

    Args:
        filename (str): Name of the file.
        reference_file (Optional[str]): Path of a reference file. If None, current directory is used.

    Returns:
        str: The absolute path of the file.
    """
    if filename.startswith("/"):
        return filename
    if reference_file:
        dir_name = os.path.dirname(reference_file)
    else:
        dir_name = os.getcwd()
    abs_dir_name = os.path.abspath(dir_name)
    abs_path = os.path.join(abs_dir_name, filename)
    return abs_path
