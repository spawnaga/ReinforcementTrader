"""
High-Performance GPU Data Pipeline for Trading System
Based on proven approach for handling large OHLCV datasets
"""

import os
import logging
import warnings
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import GPU libraries, fallback to CPU if not available
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.cuda.current_device()
            import cudf
            import cupy as cp
            # Test basic operations
            test_array = cp.array([1, 2, 3])
            _ = cp.sum(test_array)
            HAS_GPU = True
            logger.info(f"GPU processing enabled with {torch.cuda.device_count()} GPUs")
        except Exception as cuda_err:
            logger.warning(f"CUDA driver issues: {cuda_err}")
            raise ImportError("CUDA driver issues")
    else:
        raise ImportError("No GPUs available")
except (ImportError, AttributeError, RuntimeError):
    cudf = pd
    cp = np
    HAS_GPU = False
    logger.info("Using CPU fallback for data processing")

class GPUDataLoader:
    """
    High-performance data loader optimized for large OHLCV datasets
    """
    
    def __init__(self, cache_dir: str = "./data_cache", chunk_size: int = 500000):
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        os.makedirs(cache_dir, exist_ok=True)
        
    def hash_file(self, filepath: str) -> str:
        """Generate hash of file for cache validation"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def load_nq_data(self, filepath: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load NQ data with GPU acceleration and caching
        
        Args:
            filepath: Path to data file
            max_rows: Maximum rows to load (for testing)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        file_hash = self.hash_file(filepath)
        cache_file = os.path.join(self.cache_dir, f"nq_data_{file_hash}.parquet")
        
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            try:
                if HAS_GPU:
                    return cudf.read_parquet(cache_file).to_pandas()
                else:
                    return pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, reprocessing...")
        
        # Load raw data
        logger.info(f"Loading data from {filepath}")
        return self._load_and_process_file(filepath, cache_file, max_rows)
    
    def _load_and_process_file(self, filepath: str, cache_file: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load and process file with chunking for memory efficiency"""
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"Processing file: {filepath} (Size: {file_size_mb:.2f} MB)")
        
        # Detect separator by examining first line
        with open(filepath, 'r') as f:
            first_line = f.readline()
        
        # Check if comma or tab separated
        if ',' in first_line:
            sep = ','
        elif '\t' in first_line:
            sep = '\t'
        else:
            sep = ','  # Default to comma
        
        # For very large files, use chunked reading
        if file_size_mb > 100 or (max_rows and max_rows > 1000000):
            return self._load_chunked(filepath, sep, cache_file, max_rows)
        else:
            return self._load_direct(filepath, sep, cache_file, max_rows)
    
    def _load_direct(self, filepath: str, sep: str, cache_file: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Direct loading for smaller files"""
        try:
            # Try to detect headers
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
            
            has_header = any(word in first_line.lower() for word in ['date', 'time', 'open', 'high', 'low', 'close'])
            
            if HAS_GPU:
                df = cudf.read_csv(
                    filepath,
                    sep=sep,
                    header=0 if has_header else None,
                    names=None if has_header else ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                    nrows=max_rows
                )
            else:
                df = pd.read_csv(
                    filepath,
                    sep=sep,
                    header=0 if has_header else None,
                    names=None if has_header else ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                    nrows=max_rows,
                    low_memory=False
                )
            
            df = self._standardize_columns(df)
            df = self._process_data(df)
            
            # Cache the processed data
            self._save_cache(df, cache_file)
            
            # Convert to pandas if using GPU
            if HAS_GPU and hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise
    
    def _load_chunked(self, filepath: str, sep: str, cache_file: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Chunked loading for large files"""
        logger.info(f"Using chunked loading (chunk size: {self.chunk_size:,} rows)")
        
        chunks = []
        total_rows = 0
        
        try:
            # Detect headers
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
            
            has_header = any(word in first_line.lower() for word in ['date', 'time', 'open', 'high', 'low', 'close'])
            
            # Read in chunks
            for chunk_df in pd.read_csv(
                filepath,
                sep=sep,
                header=0 if has_header else None,
                names=None if has_header else ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                chunksize=self.chunk_size,
                low_memory=False
            ):
                if HAS_GPU:
                    chunk_df = cudf.from_pandas(chunk_df)
                
                chunk_df = self._standardize_columns(chunk_df)
                chunk_df = self._process_data(chunk_df)
                
                if HAS_GPU and hasattr(chunk_df, 'to_pandas'):
                    chunk_df = chunk_df.to_pandas()
                
                chunks.append(chunk_df)
                total_rows += len(chunk_df)
                
                if total_rows % (self.chunk_size * 5) == 0:
                    logger.info(f"Processed {total_rows:,} rows...")
                
                if max_rows and total_rows >= max_rows:
                    break
            
            logger.info(f"Combining {len(chunks)} chunks (total {total_rows:,} rows)")
            df = pd.concat(chunks, ignore_index=True)
            
            # Trim to max_rows if specified
            if max_rows:
                df = df.head(max_rows)
            
            # Cache the processed data
            self._save_cache(df, cache_file)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in chunked loading: {e}")
            raise
    
    def _standardize_columns(self, df):
        """Standardize column names"""
        if hasattr(df.columns, 'tolist'):
            current_cols = df.columns.tolist()
        else:
            current_cols = list(df.columns)
        
        # Map columns
        col_mapping = {}
        for i, col in enumerate(current_cols):
            col_str = str(col).lower()
            if i == 0 or 'date' in col_str or 'time' in col_str:
                col_mapping[col] = 'timestamp'
            elif i == 1 or 'open' in col_str:
                col_mapping[col] = 'open'
            elif i == 2 or 'high' in col_str:
                col_mapping[col] = 'high'
            elif i == 3 or 'low' in col_str:
                col_mapping[col] = 'low'
            elif i == 4 or 'close' in col_str:
                col_mapping[col] = 'close'
            elif i == 5 or 'vol' in col_str:
                col_mapping[col] = 'volume'
        
        df = df.rename(columns=col_mapping)
        
        # Ensure we have required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 1000  # Default volume
        
        return df
    
    def _process_data(self, df):
        """Process and clean data"""
        # Convert timestamp
        if HAS_GPU and hasattr(df, 'to_pandas'):
            # Convert to pandas for datetime processing
            temp_df = df.to_pandas()
            temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')
            df = cudf.from_pandas(temp_df)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Drop invalid rows
        df = df.dropna(subset=['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _save_cache(self, df, cache_file: str):
        """Save processed data to cache"""
        try:
            # Convert to pandas if needed
            if HAS_GPU and hasattr(df, 'to_pandas'):
                save_df = df.to_pandas()
            else:
                save_df = df
            
            save_df.to_parquet(cache_file, compression='snappy')
            logger.info(f"Cached data saved to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def clear_cache(self):
        """Clear all cached files"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            logger.info("Cache cleared")