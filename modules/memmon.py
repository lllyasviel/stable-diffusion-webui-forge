import logging
import threading
import time
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class MemUsageMonitor(threading.Thread):
    """Memory usage monitor for CUDA devices.
    
    Monitors GPU memory usage in a separate thread and provides
    statistics about memory consumption during generation.
    """
    
    def __init__(self, name: str, device: torch.device, opts: Any):
        super().__init__(name=name, daemon=True)
        self.device = device
        self.opts = opts
        self.disabled = False
        self._shutdown_event = threading.Event()
        self._run_flag = threading.Event()
        self._data: Dict[str, Any] = defaultdict(int)
        self._lock = threading.RLock()  # For thread-safe data access

        try:
            self._validate_device()
            self._test_cuda_operations()
        except Exception as e:
            logger.warning(f"Memory monitor initialization failed for device {device}: {e}")
            self.disabled = True

    def _validate_device(self) -> None:
        """Validate that the device is CUDA compatible."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        if self.device.type != 'cuda':
            raise RuntimeError(f"Device type {self.device.type} is not supported")
            
        if self.opts is None:
            raise RuntimeError("Options object is required")
            
        if not hasattr(self.opts, 'memmon_poll_rate'):
            raise RuntimeError("Options object missing memmon_poll_rate attribute")
            
        if self.opts.memmon_poll_rate < 0:
            raise ValueError("memmon_poll_rate must be non-negative")

    def _test_cuda_operations(self) -> None:
        """Test CUDA operations to ensure they work correctly."""
        self._get_cuda_memory_info()
        torch.cuda.memory_stats(self.device)

    def _get_cuda_memory_info(self) -> Tuple[int, int]:
        """Get CUDA memory information for the monitored device."""
        try:
            if self.device.index is not None:
                index = self.device.index
            else:
                index = torch.cuda.current_device()
            return torch.cuda.mem_get_info(index)
        except RuntimeError as e:
            logger.error(f"Failed to get CUDA memory info: {e}")
            raise

    def run(self) -> None:
        """Main monitoring loop."""
        if self.disabled:
            logger.debug("Memory monitor is disabled, exiting")
            return

        logger.debug(f"Starting memory monitor for device {self.device}")
        
        while not self._shutdown_event.is_set():
            # Wait for monitoring to be enabled
            self._run_flag.wait()
            
            if self._shutdown_event.is_set():
                break

            try:
                # Reset peak memory stats for this monitoring session
                torch.cuda.reset_peak_memory_stats(self.device)
                
                with self._lock:
                    self._data.clear()

                # Check if monitoring is disabled
                if self.opts.memmon_poll_rate <= 0:
                    self._run_flag.clear()
                    continue

                # Initialize minimum free memory
                with self._lock:
                    self._data["min_free"] = self._get_cuda_memory_info()[0]

                # Monitoring loop
                poll_interval = 1.0 / self.opts.memmon_poll_rate
                while self._run_flag.is_set() and not self._shutdown_event.is_set():
                    try:
                        free, total = self._get_cuda_memory_info()
                        with self._lock:
                            self._data["min_free"] = min(self._data["min_free"], free)
                        
                        time.sleep(poll_interval)
                    except Exception as e:
                        logger.error(f"Error during memory monitoring: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"Error in memory monitor run loop: {e}")
                time.sleep(1)  # Prevent tight error loop

        logger.debug(f"Memory monitor stopped for device {self.device}")

    def dump_debug(self) -> None:
        """Dump debug information about memory usage."""
        if self.disabled:
            logger.info("Memory monitor is disabled")
            return
            
        try:
            logger.info(f"Memory monitor ({self}) recorded data:")
            for k, v in self.read().items():
                # Convert to MB for readability
                mb_value = -(v // -(1024 ** 2))
                logger.info(f"  {k}: {mb_value} MB")

            logger.info(f"Raw torch memory stats for {self.device}:")
            tm = torch.cuda.memory_stats(self.device)
            for k, v in tm.items():
                if 'bytes' not in k:
                    continue
                # Convert to MB for readability
                mb_value = -(v // -(1024 ** 2))
                prefix = '\t' if 'peak' in k else ''
                logger.info(f"  {prefix}{k}: {mb_value} MB")

            logger.info("Memory summary:")
            logger.info(torch.cuda.memory_summary(self.device))
            
        except Exception as e:
            logger.error(f"Error dumping debug info: {e}")

    def monitor(self) -> None:
        """Start monitoring memory usage."""
        if not self.disabled:
            self._run_flag.set()
            logger.debug("Memory monitoring enabled")

    def read(self) -> Dict[str, Any]:
        """Read current memory statistics."""
        if self.disabled:
            return {}
            
        try:
            with self._lock:
                # Get current memory info
                free, total = self._get_cuda_memory_info()
                self._data["free"] = free
                self._data["total"] = total

                # Get torch memory stats
                torch_stats = torch.cuda.memory_stats(self.device)
                self._data["active"] = torch_stats["active.all.current"]
                self._data["active_peak"] = torch_stats["active_bytes.all.peak"]
                self._data["reserved"] = torch_stats["reserved_bytes.all.current"]
                self._data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
                
                # Calculate system peak if we have min_free data
                if "min_free" in self._data:
                    self._data["system_peak"] = total - self._data["min_free"]

                return dict(self._data)
                
        except Exception as e:
            logger.error(f"Error reading memory stats: {e}")
            return {}

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return final statistics."""
        if not self.disabled:
            self._run_flag.clear()
            logger.debug("Memory monitoring stopped")
        return self.read()

    def shutdown(self) -> None:
        """Gracefully shutdown the monitor thread."""
        if not self.disabled:
            logger.debug("Shutting down memory monitor")
            self._shutdown_event.set()
            self._run_flag.set()  # Wake up the thread if it's waiting
            
            # Wait for thread to finish with timeout
            if self.is_alive():
                self.join(timeout=5.0)
                if self.is_alive():
                    logger.warning("Memory monitor thread did not shutdown gracefully")

    def __enter__(self):
        """Context manager entry."""
        self.monitor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
