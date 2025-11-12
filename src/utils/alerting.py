"""
Alerting System for Pipeline Monitoring
=======================================
Provides automated alerting for pipeline failures, data quality issues, and performance degradation.
Supports email, Slack, and logging-based notifications.
"""
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

from src.config import DATA_QUALITY_CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Centralized alerting system for pipeline monitoring.
    """

    def __init__(self):
        self.alert_history = []
        self.alert_configs = DATA_QUALITY_CONFIG.get('alerting', {})
        self.slack_client = None

        # Setup Slack if enabled
        if self.alert_configs.get('enable_slack_alerts', False):
            self._setup_slack()

    def _setup_slack(self):
        """Setup Slack client."""
        slack_token = self._get_slack_token()
        if slack_token and SLACK_AVAILABLE:
            self.slack_client = WebClient(token=slack_token)
        else:
            logger.warning("Slack alerting enabled but token not found or SDK not available")

    def _get_slack_token(self) -> Optional[str]:
        """Get Slack token from environment or config file."""
        import os

        # Try environment variable first
        token = os.getenv('SLACK_BOT_TOKEN')

        # Try config file
        if not token:
            config_file = PROJECT_ROOT / 'config' / 'slack_config.json'
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        token = config.get('bot_token')
                except Exception as e:
                    logger.warning(f"Failed to load Slack config: {e}")

        return token

    def send_alert(self,
                  alert_type: str,
                  message: str,
                  severity: str = 'info',
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Send alert through configured channels.

        Args:
            alert_type: Type of alert (e.g., 'data_quality', 'pipeline_failure')
            message: Alert message
            severity: Alert severity ('info', 'warning', 'error', 'critical')
            metadata: Additional metadata for the alert
        """
        timestamp = datetime.now().isoformat()

        alert = {
            'timestamp': timestamp,
            'type': alert_type,
            'message': message,
            'severity': severity,
            'metadata': metadata or {}
        }

        # Store alert history
        self.alert_history.append(alert)
        if len(self.alert_history) > 1000:  # Keep last 1000 alerts
            self.alert_history = self.alert_history[-1000:]

        # Log alert
        log_message = f"[{severity.upper()}] {alert_type}: {message}"
        if severity == 'error':
            logger.error(log_message)
        elif severity == 'warning':
            logger.warning(log_message)
        else:
            logger.info(log_message)

        # Send through channels
        if severity in ['warning', 'error', 'critical']:
            self._send_email_alert(alert)
            self._send_slack_alert(alert)

        # Save alert to file
        self._save_alert_to_file(alert)

    def alert_data_quality_issue(self,
                               dataset_name: str,
                               quality_score: float,
                               issues: List[str]):
        """
        Alert for data quality issues.

        Args:
            dataset_name: Name of the dataset
            quality_score: Quality score (0-100)
            issues: List of quality issues
        """
        threshold = self.alert_configs.get('alert_on_quality_below', 70)

        if quality_score < threshold:
            message = f"Data quality alert for {dataset_name}: Score {quality_score:.1f}/100 (threshold: {threshold})"
            if issues:
                message += f"\nIssues: {', '.join(issues[:3])}"  # Show first 3 issues

            severity = 'error' if quality_score < 50 else 'warning'

            self.send_alert(
                alert_type='data_quality',
                message=message,
                severity=severity,
                metadata={
                    'dataset': dataset_name,
                    'quality_score': quality_score,
                    'issues_count': len(issues),
                    'issues': issues
                }
            )

    def alert_pipeline_failure(self,
                             pipeline_stage: str,
                             error_message: str,
                             retry_count: int = 0):
        """
        Alert for pipeline failures.

        Args:
            pipeline_stage: Stage where failure occurred
            error_message: Error message
            retry_count: Number of retries attempted
        """
        message = f"Pipeline failure in {pipeline_stage}"
        if retry_count > 0:
            message += f" (after {retry_count} retries)"

        self.send_alert(
            alert_type='pipeline_failure',
            message=message,
            severity='error',
            metadata={
                'stage': pipeline_stage,
                'error': error_message,
                'retry_count': retry_count
            }
        )

    def alert_data_drift(self,
                        dataset_name: str,
                        drift_metrics: Dict[str, Any]):
        """
        Alert for data drift detection.

        Args:
            dataset_name: Name of the dataset
            drift_metrics: Drift detection metrics
        """
        if not self.alert_configs.get('alert_on_drift_detected', True):
            return

        message = f"Data drift detected in {dataset_name}"
        drift_details = []

        for metric_name, metric_value in drift_metrics.items():
            if isinstance(metric_value, dict):
                drift_details.extend([f"{k}: {v}" for k, v in metric_value.items()])
            else:
                drift_details.append(f"{metric_name}: {metric_value}")

        if drift_details:
            message += f"\nDetails: {', '.join(drift_details[:5])}"

        self.send_alert(
            alert_type='data_drift',
            message=message,
            severity='warning',
            metadata={
                'dataset': dataset_name,
                'drift_metrics': drift_metrics
            }
        )

    def alert_performance_issue(self,
                              component: str,
                              metric_name: str,
                              current_value: float,
                              threshold: float):
        """
        Alert for performance issues.

        Args:
            component: Component with performance issue
            metric_name: Name of the metric
            current_value: Current metric value
            threshold: Threshold value
        """
        message = f"Performance alert: {component} {metric_name} = {current_value:.2f} (threshold: {threshold:.2f})"

        severity = 'warning' if current_value > threshold * 1.2 else 'info'

        self.send_alert(
            alert_type='performance',
            message=message,
            severity=severity,
            metadata={
                'component': component,
                'metric': metric_name,
                'current_value': current_value,
                'threshold': threshold
            }
        )

    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send alert via email."""
        if not self.alert_configs.get('enable_email_alerts', False):
            return

        try:
            # Email configuration
            smtp_server = self._get_email_config('smtp_server', 'smtp.gmail.com')
            smtp_port = self._get_email_config('smtp_port', 587)
            sender_email = self._get_email_config('sender_email')
            sender_password = self._get_email_config('sender_password')
            recipient_emails = self._get_email_config('recipient_emails', [])

            if not all([sender_email, sender_password, recipient_emails]):
                logger.warning("Email configuration incomplete")
                return

            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipient_emails)
            msg['Subject'] = f"[{alert['severity'].upper()}] SmartGrocy Pipeline Alert"

            body = f"""
            Alert Details:
            - Type: {alert['type']}
            - Severity: {alert['severity']}
            - Time: {alert['timestamp']}
            - Message: {alert['message']}

            Metadata: {json.dumps(alert.get('metadata', {}), indent=2)}
            """

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, recipient_emails, text)
            server.quit()

            logger.info(f"Email alert sent to {len(recipient_emails)} recipients")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _send_slack_alert(self, alert: Dict[str, Any]):
        """Send alert via Slack."""
        if not self.slack_client:
            return

        try:
            channel = self._get_slack_config('channel', '#pipeline-alerts')

            # Format message for Slack
            severity_emoji = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '❌',
                'critical': '🚨'
            }.get(alert['severity'], '📢')

            message = f"""
{severity_emoji} *SmartGrocy Pipeline Alert*

*Type:* {alert['type']}
*Severity:* {alert['severity']}
*Time:* {alert['timestamp']}

{alert['message']}
            """

            if alert.get('metadata'):
                metadata_str = "\n".join([f"• {k}: {v}" for k, v in alert['metadata'].items()])
                message += f"\n\n*Details:*\n{metadata_str}"

            self.slack_client.chat_postMessage(
                channel=channel,
                text=message,
                username='SmartGrocy Pipeline Monitor'
            )

            logger.info(f"Slack alert sent to {channel}")

        except SlackApiError as e:
            logger.error(f"Failed to send Slack alert: {e}")
        except Exception as e:
            logger.error(f"Slack alert error: {e}")

    def _save_alert_to_file(self, alert: Dict[str, Any]):
        """Save alert to file for historical tracking."""
        try:
            alerts_dir = PROJECT_ROOT / 'logs' / 'alerts'
            alerts_dir.mkdir(exist_ok=True)

            alert_file = alerts_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"

            with open(alert_file, 'a', encoding='utf-8') as f:
                json.dump(alert, f, ensure_ascii=False)
                f.write('\n')

        except Exception as e:
            logger.error(f"Failed to save alert to file: {e}")

    def _get_email_config(self, key: str, default=None) -> Any:
        """Get email configuration."""
        import os

        # Try environment variables
        env_key = f"EMAIL_{key.upper()}"
        value = os.getenv(env_key)

        # Try config file
        if value is None:
            config_file = PROJECT_ROOT / 'config' / 'email_config.json'
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        value = config.get(key, default)
                except Exception:
                    pass

        return value or default

    def _get_slack_config(self, key: str, default=None) -> Any:
        """Get Slack configuration."""
        import os

        # Try environment variables
        env_key = f"SLACK_{key.upper()}"
        value = os.getenv(env_key)

        # Try config file
        if value is None:
            config_file = PROJECT_ROOT / 'config' / 'slack_config.json'
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        value = config.get(key, default)
                except Exception:
                    pass

        return value or default

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get alert summary for the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            Alert summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]

        summary = {
            'total_alerts': len(recent_alerts),
            'by_severity': {},
            'by_type': {},
            'time_range': f"{hours} hours"
        }

        for alert in recent_alerts:
            severity = alert['severity']
            alert_type = alert['type']

            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1

        return summary


# Global instance
alert_manager = AlertManager()
