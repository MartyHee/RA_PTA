"""
I/O utilities for Douyin data project.

Handles file reading/writing for various formats (JSONL, Parquet, CSV, etc.)
with proper error handling and path management.
"""
import json
import pickle
from typing import Any, Dict, List, Optional, Union, Iterator
from pathlib import Path
import logging
import gzip
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> Any:
    """Read JSON file.

    Args:
        file_path: Path to JSON file.
        encoding: File encoding.

    Returns:
        Parsed JSON data.
    """
    file_path = Path(file_path)
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON {file_path}: {e}")
        raise


def write_json(file_path: Union[str, Path], data: Any, indent: int = 2,
               encoding: str = 'utf-8'):
    """Write JSON file.

    Args:
        file_path: Path to JSON file.
        data: Data to write.
        indent: JSON indentation.
        encoding: File encoding.
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    try:
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
        logger.debug(f"JSON written to {file_path}")
    except Exception as e:
        logger.error(f"Failed to write JSON {file_path}: {e}")
        raise


def read_jsonl(file_path: Union[str, Path], encoding: str = 'utf-8') -> List[Dict]:
    """Read JSONL file (one JSON object per line).

    Args:
        file_path: Path to JSONL file.
        encoding: File encoding.

    Returns:
        List of dictionaries.
    """
    file_path = Path(file_path)
    data = []

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_num} in {file_path}: {e}")
                        continue
        return data
    except Exception as e:
        logger.error(f"Failed to read JSONL {file_path}: {e}")
        raise


def write_jsonl(file_path: Union[str, Path], data: List[Dict], mode: str = 'w',
                encoding: str = 'utf-8'):
    """Write JSONL file.

    Args:
        file_path: Path to JSONL file.
        data: List of dictionaries to write.
        mode: Write mode ('w' for write, 'a' for append).
        encoding: File encoding.
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    try:
        with open(file_path, mode, encoding=encoding) as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + '\n')
        logger.debug(f"JSONL written to {file_path} (mode: {mode})")
    except Exception as e:
        logger.error(f"Failed to write JSONL {file_path}: {e}")
        raise


def read_parquet(file_path: Union[str, Path]) -> pd.DataFrame:
    """Read Parquet file.

    Args:
        file_path: Path to Parquet file.

    Returns:
        DataFrame.
    """
    file_path = Path(file_path)
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Failed to read Parquet {file_path}: {e}")
        raise


def write_parquet(file_path: Union[str, Path], data: Union[pd.DataFrame, List[Dict]],
                  mode: str = 'w', **kwargs):
    """Write Parquet file.

    Args:
        file_path: Path to Parquet file.
        data: DataFrame or list of dictionaries.
        mode: Write mode ('w' for write, 'a' for append).
        **kwargs: Additional arguments to pd.DataFrame.to_parquet.
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    # Convert list of dicts to DataFrame if needed
    if isinstance(data, list):
        data = pd.DataFrame(data)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame or list, got {type(data)}")

    try:
        if mode == 'a' and file_path.exists():
            # Read existing data
            existing_df = pd.read_parquet(file_path)
            # Append new data
            combined_df = pd.concat([existing_df, data], ignore_index=True)
            combined_df.to_parquet(file_path, **kwargs)
        else:
            data.to_parquet(file_path, **kwargs)

        logger.debug(f"Parquet written to {file_path} (mode: {mode}, rows: {len(data)})")
    except Exception as e:
        logger.error(f"Failed to write Parquet {file_path}: {e}")
        raise


def read_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read CSV file.

    Args:
        file_path: Path to CSV file.
        **kwargs: Additional arguments to pd.read_csv.

    Returns:
        DataFrame.
    """
    file_path = Path(file_path)
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Failed to read CSV {file_path}: {e}")
        raise


def write_csv(file_path: Union[str, Path], data: Union[pd.DataFrame, List[Dict]],
              mode: str = 'w', **kwargs):
    """Write CSV file.

    Args:
        file_path: Path to CSV file.
        data: DataFrame or list of dictionaries.
        mode: Write mode ('w' for write, 'a' for append).
        **kwargs: Additional arguments to pd.DataFrame.to_csv.
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    # Convert list of dicts to DataFrame if needed
    if isinstance(data, list):
        data = pd.DataFrame(data)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame or list, got {type(data)}")

    try:
        if mode == 'a' and file_path.exists():
            # Append mode
            header = False
        else:
            header = True

        data.to_csv(file_path, mode=mode, header=header, **kwargs)
        logger.debug(f"CSV written to {file_path} (mode: {mode}, rows: {len(data)})")
    except Exception as e:
        logger.error(f"Failed to write CSV {file_path}: {e}")
        raise


def read_pickle(file_path: Union[str, Path]) -> Any:
    """Read pickle file.

    Args:
        file_path: Path to pickle file.

    Returns:
        Unpickled data.
    """
    file_path = Path(file_path)
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to read pickle {file_path}: {e}")
        raise


def write_pickle(file_path: Union[str, Path], data: Any):
    """Write pickle file.

    Args:
        file_path: Path to pickle file.
        data: Data to pickle.
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Pickle written to {file_path}")
    except Exception as e:
        logger.error(f"Failed to write pickle {file_path}: {e}")
        raise


def compress_file(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None,
                  remove_original: bool = False):
    """Compress file with gzip.

    Args:
        input_path: Path to input file.
        output_path: Path to output gzip file. If None, adds .gz extension.
        remove_original: Whether to remove original file after compression.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(input_path.suffix + '.gz')
    else:
        output_path = Path(output_path)

    ensure_dir(output_path.parent)

    try:
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.debug(f"Compressed {input_path} -> {output_path}")

        if remove_original:
            input_path.unlink()
            logger.debug(f"Removed original file: {input_path}")

    except Exception as e:
        logger.error(f"Failed to compress {input_path}: {e}")
        raise


def decompress_file(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None,
                    remove_compressed: bool = False):
    """Decompress gzip file.

    Args:
        input_path: Path to gzip file.
        output_path: Path to output file. If None, removes .gz extension.
        remove_compressed: Whether to remove compressed file after decompression.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('')  # Remove .gz
    else:
        output_path = Path(output_path)

    ensure_dir(output_path.parent)

    try:
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.debug(f"Decompressed {input_path} -> {output_path}")

        if remove_compressed:
            input_path.unlink()
            logger.debug(f"Removed compressed file: {input_path}")

    except Exception as e:
        logger.error(f"Failed to decompress {input_path}: {e}")
        raise


def list_files(dir_path: Union[str, Path], pattern: str = '*',
               recursive: bool = False) -> List[Path]:
    """List files in directory.

    Args:
        dir_path: Directory path.
        pattern: Glob pattern.
        recursive: Whether to search recursively.

    Returns:
        List of file paths.
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {dir_path}")
        return []

    if recursive:
        return list(dir_path.rglob(pattern))
    else:
        return list(dir_path.glob(pattern))


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get file information.

    Args:
        file_path: File path.

    Returns:
        Dictionary with file info.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {'exists': False}

    stats = file_path.stat()
    return {
        'exists': True,
        'path': str(file_path),
        'name': file_path.name,
        'size_bytes': stats.st_size,
        'size_mb': stats.st_size / (1024 * 1024),
        'created': stats.st_ctime,
        'modified': stats.st_mtime,
        'accessed': stats.st_atime,
        'is_file': file_path.is_file(),
        'is_dir': file_path.is_dir(),
        'extension': file_path.suffix,
        'parent': str(file_path.parent)
    }


def save_figure(fig, file_path: Union[str, Path], dpi: int = 300, **kwargs):
    """Save matplotlib figure.

    Args:
        fig: Matplotlib figure.
        file_path: Output file path.
        dpi: DPI for saving.
        **kwargs: Additional arguments to fig.savefig.
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    try:
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight', **kwargs)
        logger.debug(f"Figure saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save figure {file_path}: {e}")
        raise


def read_text(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """Read text file.

    Args:
        file_path: Path to text file.
        encoding: File encoding.

    Returns:
        File content as string.
    """
    file_path = Path(file_path)
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read text file {file_path}: {e}")
        raise


def write_text(file_path: Union[str, Path], text: str, mode: str = 'w',
               encoding: str = 'utf-8'):
    """Write text file.

    Args:
        file_path: Path to text file.
        text: Text to write.
        mode: Write mode ('w' for write, 'a' for append).
        encoding: File encoding.
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    try:
        with open(file_path, mode, encoding=encoding) as f:
            f.write(text)
        logger.debug(f"Text written to {file_path} (mode: {mode}, length: {len(text)})")
    except Exception as e:
        logger.error(f"Failed to write text file {file_path}: {e}")
        raise


# Batch operations
def batch_read_jsonl(file_path: Union[str, Path], batch_size: int = 1000,
                     encoding: str = 'utf-8') -> Iterator[List[Dict]]:
    """Read JSONL file in batches.

    Args:
        file_path: Path to JSONL file.
        batch_size: Number of lines per batch.
        encoding: File encoding.

    Yields:
        Batches of dictionaries.
    """
    file_path = Path(file_path)
    batch = []

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        batch.append(json.loads(line))
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON line: {e}")
                        continue

            # Yield remaining items
            if batch:
                yield batch

    except Exception as e:
        logger.error(f"Failed to read JSONL {file_path}: {e}")
        raise


def batch_write_jsonl(file_path: Union[str, Path], data_generator: Iterator[List[Dict]],
                      mode: str = 'w', encoding: str = 'utf-8'):
    """Write JSONL file in batches.

    Args:
        file_path: Path to JSONL file.
        data_generator: Generator yielding batches of dictionaries.
        mode: Write mode.
        encoding: File encoding.
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    try:
        with open(file_path, mode, encoding=encoding) as f:
            for batch_num, batch in enumerate(data_generator, 1):
                for item in batch:
                    f.write(json.dumps(item, ensure_ascii=False, default=str) + '\n')
                logger.debug(f"Batch {batch_num} written ({len(batch)} items)")
    except Exception as e:
        logger.error(f"Failed to write JSONL {file_path}: {e}")
        raise