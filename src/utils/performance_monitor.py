"""
Performance Monitoring Module
============================
Monitors pipeline performance, identifies bottlenecks, and provides optimization recommendations.
"""
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import psutil

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors pipeline performance and resource usage.
    """

    def __init__(self):
        self.metrics_history = []
        self.current_session = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.session_start_time = None

    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.session_start_time = time.time()
        self.current_session = []

        # Start background monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("Performance monitoring started")

    def stop_monitoring(self) -> dict[str, Any]:
        """Stop performance monitoring and return session summary."""
        if not self.monitoring_active:
            return {}

        self.monitoring_active = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        time.time() - self.session_start_time
        session_summary = self._analyze_session()

        # Save session data
        self._save_session_data(session_summary)

        logger.info("Performance monitoring stopped")
        return session_summary

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                metrics['timestamp'] = datetime.now().isoformat()
                self.current_session.append(metrics)

                time.sleep(1)  # Collect metrics every second

            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                break

    def _collect_system_metrics(self) -> dict[str, Any]:
        """Collect current system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_read_mb = disk_io.read_bytes / (1024**2)
                disk_write_mb = disk_io.write_bytes / (1024**2)
            else:
                disk_read_mb = disk_write_mb = 0

            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                net_sent_mb = net_io.bytes_sent / (1024**2)
                net_recv_mb = net_io.bytes_recv / (1024**2)
            else:
                net_sent_mb = net_recv_mb = 0

            metrics = {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'cpu_freq_mhz': cpu_freq.current if cpu_freq else 0,
                'memory_percent': memory_percent,
                'memory_used_gb': memory_used_gb,
                'memory_available_gb': memory_available_gb,
                'disk_read_mb': disk_read_mb,
                'disk_write_mb': disk_write_mb,
                'net_sent_mb': net_sent_mb,
                'net_recv_mb': net_recv_mb
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}

    @contextmanager
    def time_operation(self, operation_name: str, metadata: dict[str, Any] | None = None):
        """
        Context manager to time an operation.

        Args:
            operation_name: Name of the operation
            metadata: Additional metadata
        """
        start_time = time.time()
        start_metrics = self._collect_system_metrics()

        try:
            yield
        finally:
            end_time = time.time()
            end_metrics = self._collect_system_metrics()

            duration = end_time - start_time

            operation_record = {
                'operation_name': operation_name,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'duration_seconds': duration,
                'start_metrics': start_metrics,
                'end_metrics': end_metrics,
                'metadata': metadata or {}
            }

            self.current_session.append(operation_record)
            logger.debug(f"Timed operation '{operation_name}': {duration:.2f}s")

    def record_operation(self, operation_name: str, duration: float, metadata: dict[str, Any] | None = None):
        """
        Record an operation manually.

        Args:
            operation_name: Name of the operation
            duration: Duration in seconds
            metadata: Additional metadata
        """
        operation_record = {
            'operation_name': operation_name,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'metadata': metadata or {}
        }

        self.current_session.append(operation_record)

    def _analyze_session(self) -> dict[str, Any]:
        """Analyze the current monitoring session."""
        if not self.current_session:
            return {}

        session_df = pd.DataFrame(self.current_session)

        # Filter out operation records for system metrics analysis
        system_metrics = session_df[session_df['timestamp'].notna() & session_df['cpu_percent'].notna()]

        analysis = {
            'session_duration': time.time() - self.session_start_time,
            'total_measurements': len(system_metrics),
            'operations_count': len(session_df) - len(system_metrics)
        }

        if not system_metrics.empty:
            # CPU analysis
            analysis['cpu_avg_percent'] = system_metrics['cpu_percent'].mean()
            analysis['cpu_max_percent'] = system_metrics['cpu_percent'].max()
            analysis['cpu_std_percent'] = system_metrics['cpu_percent'].std()

            # Memory analysis
            analysis['memory_avg_percent'] = system_metrics['memory_percent'].mean()
            analysis['memory_max_percent'] = system_metrics['memory_percent'].max()
            analysis['memory_avg_used_gb'] = system_metrics['memory_used_gb'].mean()

            # Performance bottlenecks
            analysis['bottlenecks'] = self._identify_bottlenecks(system_metrics)

        # Operation analysis
        operations = session_df[session_df['operation_name'].notna()]
        if not operations.empty:
            analysis['operation_stats'] = self._analyze_operations(operations)

        return analysis

    def _identify_bottlenecks(self, metrics_df: pd.DataFrame) -> list[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # High CPU usage
        if metrics_df['cpu_percent'].mean() > 80:
            bottlenecks.append("High CPU usage detected")

        # High memory usage
        if metrics_df['memory_percent'].mean() > 85:
            bottlenecks.append("High memory usage detected")

        # Memory spikes
        memory_std = metrics_df['memory_percent'].std()
        if memory_std > 10:
            bottlenecks.append("Memory usage spikes detected")

        # CPU frequency drops (throttling)
        if 'cpu_freq_mhz' in metrics_df.columns:
            freq_changes = metrics_df['cpu_freq_mhz'].pct_change().abs()
            if freq_changes.mean() > 0.1:
                bottlenecks.append("CPU frequency changes indicate possible throttling")

        return bottlenecks

    def _analyze_operations(self, operations_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze operation performance."""
        stats = {}

        # Group by operation name
        operation_groups = operations_df.groupby('operation_name')

        for op_name, group in operation_groups:
            durations = group['duration_seconds']
            stats[op_name] = {
                'count': len(group),
                'total_time': durations.sum(),
                'avg_time': durations.mean(),
                'max_time': durations.max(),
                'min_time': durations.min(),
                'std_time': durations.std()
            }

        # Find slowest operations
        if stats:
            slowest_op = max(stats.items(), key=lambda x: x[1]['avg_time'])
            stats['_slowest_operation'] = {
                'name': slowest_op[0],
                'avg_time': slowest_op[1]['avg_time']
            }

        return stats

    def _save_session_data(self, session_summary: dict[str, Any]):
        """Save session data to disk."""
        try:
            perf_dir = PROJECT_ROOT / 'reports' / 'performance'
            perf_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save detailed session data
            session_file = perf_dir / f'perf_session_{timestamp}.json'
            session_data = {
                'session_summary': session_summary,
                'raw_metrics': self.current_session,
                'timestamp': timestamp
            }

            import json
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)

            # Save summary to history
            self.metrics_history.append(session_summary)
            if len(self.metrics_history) > 100:  # Keep last 100 sessions
                self.metrics_history = self.metrics_history[-100:]

            logger.info(f"Performance data saved to: {session_file}")

        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")

    def get_performance_recommendations(self) -> list[str]:
        """
        Generate performance optimization recommendations.

        Returns:
            List of recommendations
        """
        recommendations = []

        if not self.metrics_history:
            return ["No performance history available"]

        # Analyze recent sessions
        recent_sessions = self.metrics_history[-5:]  # Last 5 sessions

        avg_cpu = np.mean([s.get('cpu_avg_percent', 0) for s in recent_sessions])
        avg_memory = np.mean([s.get('memory_avg_percent', 0) for s in recent_sessions])

        if avg_cpu > 90:
            recommendations.append("Consider increasing CPU cores or optimizing CPU-intensive operations")
        elif avg_cpu > 70:
            recommendations.append("Monitor CPU usage - consider parallelization if not already implemented")

        if avg_memory > 90:
            recommendations.append("High memory usage detected - consider memory optimization or more RAM")
        elif avg_memory > 80:
            recommendations.append("Memory usage is high - monitor for potential leaks")

        # Check for bottlenecks across sessions
        all_bottlenecks = []
        for session in recent_sessions:
            all_bottlenecks.extend(session.get('bottlenecks', []))

        if all_bottlenecks:
            bottleneck_counts = pd.Series(all_bottlenecks).value_counts()
            for bottleneck, count in bottleneck_counts.items():
                if count >= 3:  # Appears in 3+ sessions
                    recommendations.append(f"Recurring bottleneck: {bottleneck}")

        # Operation-specific recommendations
        for session in recent_sessions:
            op_stats = session.get('operation_stats', {})
            if '_slowest_operation' in op_stats:
                slow_op = op_stats['_slowest_operation']
                if slow_op['avg_time'] > 300:  # More than 5 minutes
                    recommendations.append(f"Optimize slow operation: {slow_op['name']} ({slow_op['avg_time']:.1f}s avg)")

        if not recommendations:
            recommendations.append("Performance looks good - no major issues detected")

        return recommendations

    def get_system_info(self) -> dict[str, Any]:
        """
        Get system information.

        Returns:
            System information dictionary
        """
        try:
            system_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'platform': psutil.platform(),
                'python_version': __import__('sys').version
            }

            # CPU frequency info
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                system_info['cpu_freq_min'] = cpu_freq.min
                system_info['cpu_freq_max'] = cpu_freq.max

            return system_info

        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}


# Global instance
performance_monitor = PerformanceMonitor()
