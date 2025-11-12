"""
Caching Layer for Pipeline Optimization
=======================================
Provides intelligent caching for expensive operations with disk persistence.
Supports incremental processing and memory-efficient operations.
"""
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List
from functools import wraps
import pandas as pd
import diskcache as dc

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


class PipelineCache:
    """
    Intelligent caching system for pipeline operations.
    """

    def __init__(self, cache_dir: Optional[Path] = None, max_size_gb: float = 10.0):
        """
        Initialize cache system.

        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = cache_dir or (PROJECT_ROOT / 'cache')
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize disk cache
        self.cache = dc.Cache(
            directory=str(self.cache_dir),
            size_limit=int(max_size_gb * 1024**3)  # Convert GB to bytes
        )

        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0
        }

        logger.info(f"Pipeline cache initialized at: {self.cache_dir}")

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """
        Generate cache key from function call signature.

        Args:
            func_name: Function name
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Create hash of function signature
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }

        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Any:
        """
        Retrieve item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found
        """
        try:
            result = self.cache.get(key)
            if result is not None:
                self.stats['hits'] += 1
                logger.debug(f"Cache hit: {key}")
                return result
            else:
                self.stats['misses'] += 1
                logger.debug(f"Cache miss: {key}")
                return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: Any, expire: Optional[int] = None):
        """
        Store item in cache.

        Args:
            key: Cache key
            value: Item to cache
            expire: Expiration time in seconds (optional)
        """
        try:
            self.cache.set(key, value, expire=expire)
            self.stats['saves'] += 1
            logger.debug(f"Cache set: {key}")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def cached_operation(self, expire_hours: int = 24):
        """
        Decorator for caching function results.

        Args:
            expire_hours: Cache expiration time in hours

        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                key = self._generate_key(func.__name__, args, kwargs)

                # Try to get from cache
                cached_result = self.get(key)
                if cached_result is not None:
                    return cached_result

                # Execute function
                logger.debug(f"Executing {func.__name__} (cache miss)")
                result = func(*args, **kwargs)

                # Cache result
                expire_seconds = expire_hours * 3600
                self.set(key, result, expire=expire_seconds)

                return result

            return wrapper
        return decorator

    def clear_expired(self):
        """Clear expired cache entries."""
        try:
            self.cache.expire()
            logger.info("Expired cache entries cleared")
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")

    def clear_all(self):
        """Clear all cache entries."""
        try:
            self.cache.clear()
            self.stats = {'hits': 0, 'misses': 0, 'saves': 0}
            logger.info("All cache entries cleared")
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_info = {
                'cache_size_mb': len(self.cache) / 1024**2,
                'cache_dir': str(self.cache_dir),
                'stats': self.stats.copy(),
                'hit_rate': self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1)
            }
            return cache_info
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {'error': str(e)}


class IncrementalProcessor:
    """
    Incremental processing for large datasets.
    """

    def __init__(self, cache: PipelineCache):
        """
        Initialize incremental processor.

        Args:
            cache: Pipeline cache instance
        """
        self.cache = cache
        self.processing_state = {}

    def get_processing_state(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get processing state for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Processing state dictionary
        """
        state_key = f"processing_state_{dataset_name}"
        state = self.cache.get(state_key)

        if state is None:
            state = {
                'last_processed_timestamp': None,
                'processed_rows': 0,
                'chunks_processed': 0,
                'last_chunk_hash': None
            }

        return state

    def update_processing_state(self, dataset_name: str, updates: Dict[str, Any]):
        """
        Update processing state for a dataset.

        Args:
            dataset_name: Name of the dataset
            updates: State updates
        """
        state_key = f"processing_state_{dataset_name}"
        current_state = self.get_processing_state(dataset_name)
        current_state.update(updates)
        self.cache.set(state_key, current_state)

    def process_incremental(self,
                          df: pd.DataFrame,
                          dataset_name: str,
                          chunk_size: int = 10000,
                          force_reprocess: bool = False) -> pd.DataFrame:
        """
        Process dataframe incrementally with intelligent caching.

        Args:
            df: DataFrame to process
            dataset_name: Name of the dataset
            chunk_size: Size of processing chunks
            force_reprocess: Force full reprocessing

        Returns:
            Processed DataFrame
        """
        logger.info(f"Starting incremental processing for {dataset_name}")

        state = self.get_processing_state(dataset_name)

        # Check if full reprocessing is needed
        current_hash = hashlib.md5(str(df.shape + tuple(df.dtypes)).encode()).hexdigest()

        if force_reprocess or state['last_chunk_hash'] != current_hash:
            logger.info("Data changed or forced reprocess, full reprocessing required")
            # Clear old chunks
            self._clear_old_chunks(dataset_name)
            state = {
                'last_processed_timestamp': None,
                'processed_rows': 0,
                'chunks_processed': 0,
                'last_chunk_hash': current_hash,
                'total_chunks': 0
            }

        # Process in chunks with progress tracking
        results = []
        total_chunks = len(df) // chunk_size + 1
        cached_chunks = 0
        processed_chunks = 0

        logger.info(f"Processing {total_chunks} chunks (chunk_size={chunk_size})")

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_id = f"{dataset_name}_chunk_{i//chunk_size}"
            chunk_idx = i // chunk_size + 1

            # Check if chunk already processed and valid
            cached_chunk = self.cache.get(chunk_id)
            if cached_chunk is not None and not force_reprocess:
                results.append(cached_chunk)
                cached_chunks += 1
                logger.debug(f"Using cached chunk {chunk_idx}/{total_chunks}")
            else:
                # Process chunk
                try:
                    processed_chunk = self._process_chunk(chunk, chunk_id)
                    results.append(processed_chunk)
                    processed_chunks += 1

                    # Cache processed chunk with metadata
                    chunk_metadata = {
                        'data': processed_chunk,
                        'chunk_index': chunk_idx,
                        'processed_at': pd.Timestamp.now().isoformat(),
                        'original_shape': chunk.shape,
                        'processed_shape': processed_chunk.shape
                    }
                    self.cache.set(chunk_id, chunk_metadata)

                    logger.debug(f"Processed chunk {chunk_idx}/{total_chunks}")

                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk_idx}: {e}")
                    # Continue with other chunks but mark failure
                    continue

        # Combine results
        if results:
            try:
                final_result = pd.concat(results, ignore_index=True)

                # Validate final result
                if len(final_result) != len(df):
                    logger.warning(f"Row count mismatch: expected {len(df)}, got {len(final_result)}")

                # Update state
                self.update_processing_state(dataset_name, {
                    'processed_rows': len(final_result),
                    'chunks_processed': len(results),
                    'cached_chunks': cached_chunks,
                    'newly_processed_chunks': processed_chunks,
                    'last_processed_timestamp': pd.Timestamp.now().isoformat(),
                    'total_chunks': total_chunks,
                    'cache_hit_rate': cached_chunks / total_chunks if total_chunks > 0 else 0
                })

                logger.info(f"Incremental processing complete: {len(final_result)} rows")
                logger.info(f"Cache hit rate: {cached_chunks}/{total_chunks} chunks ({cached_chunks/total_chunks*100:.1f}%)")

                return final_result

            except Exception as e:
                logger.error(f"Failed to combine processed chunks: {e}")
                return df.copy()
        else:
            logger.warning("No chunks were successfully processed")
            return df.copy()

    def _clear_old_chunks(self, dataset_name: str):
        """Clear old cached chunks for a dataset."""
        try:
            # Find all chunk keys for this dataset
            chunk_keys = []
            for key in self.cache.iterkeys():
                if key.startswith(f"{dataset_name}_chunk_"):
                    chunk_keys.append(key)

            # Remove old chunks
            for key in chunk_keys:
                del self.cache[key]

            logger.info(f"Cleared {len(chunk_keys)} old chunks for {dataset_name}")

        except Exception as e:
            logger.warning(f"Failed to clear old chunks: {e}")

    def get_incremental_stats(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get incremental processing statistics for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Processing statistics
        """
        state = self.get_processing_state(dataset_name)

        # Calculate additional metrics
        stats = {
            'dataset_name': dataset_name,
            'total_rows_processed': state.get('processed_rows', 0),
            'chunks_processed': state.get('chunks_processed', 0),
            'cached_chunks_used': state.get('cached_chunks', 0),
            'new_chunks_processed': state.get('newly_processed_chunks', 0),
            'cache_hit_rate': state.get('cache_hit_rate', 0),
            'last_processed': state.get('last_processed_timestamp'),
            'data_fingerprint': state.get('last_chunk_hash', 'none')
        }

        return stats

    def _process_chunk(self, chunk: pd.DataFrame, chunk_id: str) -> pd.DataFrame:
        """
        Process a single chunk (placeholder for actual processing logic).

        Args:
            chunk: DataFrame chunk
            chunk_id: Unique chunk identifier

        Returns:
            Processed chunk
        """
        # This would be overridden by specific processing functions
        # For now, just return the chunk as-is
        return chunk


# Global instances
pipeline_cache = PipelineCache()
incremental_processor = IncrementalProcessor(pipeline_cache)
