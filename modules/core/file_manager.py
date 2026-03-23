import os
from datetime import datetime
from typing import List, Dict, Union

# Configuration for raw data storage
UPLOADS_DIR = "data/uploads"

def ensure_storage() -> None:
    """Ensures the data/uploads directory exists."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)


def save_file(uploaded_file) -> str:
    """
    Saves an uploaded CSV file to local storage.
    
    Args:
        uploaded_file: The file object from st.file_uploader.
        
    Returns:
        str: The absolute path to the saved file.
    """
    ensure_storage()
    file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def get_data_library() -> List[Dict[str, Union[str, float]]]:
    """
    Retrieves a list of available CSV files in the storage.
    
    Returns:
        List[Dict]: A list of dictionaries containing file metadata 
                    (name, size, date, path).
    """
    ensure_storage()
    files = []
    
    # Iterate through files in the directory
    for f in os.listdir(UPLOADS_DIR):
        if f.endswith('.csv'):
            path = os.path.join(UPLOADS_DIR, f)
            try:
                stats = os.stat(path)
                files.append({
                    "name": f,
                    "size": f"{stats.st_size / 1024:.1f} KB",
                    "date": datetime.fromtimestamp(stats.st_mtime).strftime("%H:%M | %d/%m"),
                    "path": path
                })
            except OSError:
                continue # Skip files we can't read
                
    return files


def delete_data(filename: str) -> bool:
    """
    Permanently deletes a file from the system.
    
    Args:
        filename (str): The name of the file to delete.
        
    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    path = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(path):
        try:
            os.remove(path)
            return True
        except OSError:
            return False
    return False


def save_dataframe(df, filename: str) -> str:
    """
    Saves a pandas DataFrame as a CSV file to the uploads directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the file (e.g., "data_cleaned.csv").

    Returns:
        str: The absolute path to the saved file.
    """
    ensure_storage()
    file_path = os.path.join(UPLOADS_DIR, filename)
    df.to_csv(file_path, index=False)
    return file_path


def save_with_auto_increment(
    df,
    base_stem: str,
    suffix: str,
) -> tuple:
    """Save a DataFrame with auto-incrementing filename to avoid overwrites.

    When ``<base_stem><suffix>.csv`` already exists, appends ``_1``, ``_2``, …
    until a free filename is found.  Returns both the final filename and the
    CSV bytes (UTF-8 encoded) for downstream use (e.g. ZIP download).

    Args:
        df:        DataFrame to save.
        base_stem: Stem of the original file (e.g. ``"employee_data"``).
        suffix:    Descriptive suffix (e.g. ``"_cleaned"``, ``"_encoded"``).

    Returns:
        Tuple of ``(filename, csv_bytes)`` where *filename* is the basename
        written to ``UPLOADS_DIR`` and *csv_bytes* is the UTF-8 encoded CSV
        content as ``bytes``.
    """
    filename = f"{base_stem}{suffix}.csv"
    counter = 1
    while os.path.exists(os.path.join(UPLOADS_DIR, filename)):
        filename = f"{base_stem}{suffix}_{counter}.csv"
        counter += 1
    save_dataframe(df, filename)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return filename, csv_bytes

