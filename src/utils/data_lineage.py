"""
Data Lineage Tracking Module
===========================
Tracks data flow through the pipeline for transparency and debugging.
Provides lineage graphs and dependency tracking.
"""
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class DataArtifact:
    """Represents a data artifact in the pipeline."""
    name: str
    artifact_type: str  # 'raw_data', 'processed_data', 'model', 'feature_table', etc.
    path: str | None = None
    shape: tuple | None = None
    schema: dict[str, str] | None = None
    created_at: str | None = None
    created_by: str | None = None
    quality_score: float | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineStep:
    """Represents a step in the pipeline."""
    step_name: str
    step_type: str  # 'load', 'transform', 'feature_engineering', 'train', 'predict'
    inputs: list[str]  # Names of input artifacts
    outputs: list[str]  # Names of output artifacts
    parameters: dict[str, Any] | None = None
    execution_time: float | None = None
    status: str | None = None  # 'success', 'failed', 'running'
    error_message: str | None = None
    executed_at: str | None = None
    executed_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataLineageTracker:
    """
    Tracks data lineage through the entire pipeline.
    """

    def __init__(self, lineage_dir: Path | None = None):
        """
        Initialize lineage tracker.

        Args:
            lineage_dir: Directory to store lineage data
        """
        self.lineage_dir = lineage_dir or (PROJECT_ROOT / 'reports' / 'lineage')
        self.lineage_dir.mkdir(exist_ok=True)

        # In-memory storage
        self.artifacts: dict[str, DataArtifact] = {}
        self.steps: list[PipelineStep] = []
        self.lineage_graph = nx.DiGraph()

        # Load existing lineage if available
        self._load_lineage()

        logger.info(f"Data lineage tracker initialized at: {self.lineage_dir}")

    def register_artifact(self, artifact: DataArtifact):
        """
        Register a data artifact.

        Args:
            artifact: DataArtifact to register
        """
        if not artifact.created_at:
            artifact.created_at = datetime.now().isoformat()

        self.artifacts[artifact.name] = artifact

        # Add to graph
        self.lineage_graph.add_node(
            artifact.name,
            type=artifact.artifact_type,
            **artifact.to_dict()
        )

        logger.debug(f"Registered artifact: {artifact.name}")

    def record_step(self, step: PipelineStep):
        """
        Record a pipeline step execution.

        Args:
            step: PipelineStep to record
        """
        if not step.executed_at:
            step.executed_at = datetime.now().isoformat()

        self.steps.append(step)

        # Add step to graph
        step_id = f"step_{len(self.steps)}"
        self.lineage_graph.add_node(
            step_id,
            node_type='step',
            **step.to_dict()
        )

        # Add edges from inputs to step
        for input_name in step.inputs:
            if input_name in self.artifacts:
                self.lineage_graph.add_edge(input_name, step_id)

        # Add edges from step to outputs
        for output_name in step.outputs:
            self.lineage_graph.add_edge(step_id, output_name)

        logger.debug(f"Recorded step: {step.step_name}")

    def get_artifact_lineage(self, artifact_name: str) -> dict[str, Any]:
        """
        Get the lineage information for an artifact.

        Args:
            artifact_name: Name of the artifact

        Returns:
            Lineage information
        """
        if artifact_name not in self.artifacts:
            return {}

        lineage = {
            'artifact': self.artifacts[artifact_name].to_dict(),
            'upstream_steps': [],
            'downstream_steps': [],
            'upstream_artifacts': [],
            'downstream_artifacts': []
        }

        # Find upstream and downstream
        if artifact_name in self.lineage_graph:
            # Upstream (predecessors)
            upstream = list(self.lineage_graph.predecessors(artifact_name))
            for node in upstream:
                node_data = self.lineage_graph.nodes[node]
                if node_data.get('node_type') == 'step':
                    lineage['upstream_steps'].append(node_data)
                else:
                    lineage['upstream_artifacts'].append(node_data)

            # Downstream (successors)
            downstream = list(self.lineage_graph.successors(artifact_name))
            for node in downstream:
                node_data = self.lineage_graph.nodes[node]
                if node_data.get('node_type') == 'step':
                    lineage['downstream_steps'].append(node_data)
                else:
                    lineage['downstream_artifacts'].append(node_data)

        return lineage

    def get_step_lineage(self, step_index: int) -> dict[str, Any]:
        """
        Get lineage information for a pipeline step.

        Args:
            step_index: Index of the step

        Returns:
            Step lineage information
        """
        if step_index >= len(self.steps):
            return {}

        step = self.steps[step_index]
        step_id = f"step_{step_index + 1}"

        lineage = {
            'step': step.to_dict(),
            'input_artifacts': [],
            'output_artifacts': []
        }

        if step_id in self.lineage_graph:
            # Input artifacts
            predecessors = list(self.lineage_graph.predecessors(step_id))
            for pred in predecessors:
                if pred in self.artifacts:
                    lineage['input_artifacts'].append(self.artifacts[pred].to_dict())

            # Output artifacts
            successors = list(self.lineage_graph.successors(step_id))
            for succ in successors:
                if succ in self.artifacts:
                    lineage['output_artifacts'].append(self.artifacts[succ].to_dict())

        return lineage

    def generate_lineage_report(self) -> dict[str, Any]:
        """
        Generate a comprehensive lineage report.

        Returns:
            Lineage report dictionary
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_artifacts': len(self.artifacts),
            'total_steps': len(self.steps),
            'artifacts_by_type': {},
            'steps_by_type': {},
            'pipeline_flow': []
        }

        # Count artifacts by type
        for artifact in self.artifacts.values():
            art_type = artifact.artifact_type
            report['artifacts_by_type'][art_type] = report['artifacts_by_type'].get(art_type, 0) + 1

        # Count steps by type
        for step in self.steps:
            step_type = step.step_type
            report['steps_by_type'][step_type] = report['steps_by_type'].get(step_type, 0) + 1

        # Pipeline flow summary
        for i, step in enumerate(self.steps):
            flow_entry = {
                'step_index': i + 1,
                'step_name': step.step_name,
                'step_type': step.step_type,
                'inputs': step.inputs,
                'outputs': step.outputs,
                'status': step.status,
                'execution_time': step.execution_time
            }
            report['pipeline_flow'].append(flow_entry)

        return report

    def export_lineage_graph(self, output_path: Path | None = None) -> str:
        """
        Export lineage graph to GraphML format.

        Args:
            output_path: Path to save the graph

        Returns:
            Path to the exported graph
        """
        if output_path is None:
            output_path = self.lineage_dir / f'lineage_graph_{datetime.now().strftime("%Y%m%d_%H%M%S")}.graphml'

        try:
            nx.write_graphml(self.lineage_graph, str(output_path))
            logger.info(f"Lineage graph exported to: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to export lineage graph: {e}")
            return ""

    def find_data_dependencies(self, artifact_name: str) -> list[str]:
        """
        Find all data dependencies for an artifact.

        Args:
            artifact_name: Name of the artifact

        Returns:
            List of dependency artifact names
        """
        if artifact_name not in self.lineage_graph:
            return []

        # Get all upstream artifacts
        dependencies = set()

        def collect_upstream(node):
            for pred in self.lineage_graph.predecessors(node):
                node_data = self.lineage_graph.nodes[pred]
                if node_data.get('node_type') != 'step':
                    dependencies.add(pred)
                    collect_upstream(pred)

        collect_upstream(artifact_name)

        return list(dependencies)

    def detect_lineage_breaks(self) -> list[str]:
        """
        Detect breaks in the data lineage.

        Returns:
            List of detected issues
        """
        issues = []

        # Check for missing artifacts
        for step in self.steps:
            for input_name in step.inputs:
                if input_name not in self.artifacts:
                    issues.append(f"Step '{step.step_name}' references missing input: {input_name}")

            for output_name in step.outputs:
                if output_name not in self.artifacts:
                    issues.append(f"Step '{step.step_name}' references missing output: {output_name}")

        # Check for orphaned artifacts
        referenced_artifacts = set()
        for step in self.steps:
            referenced_artifacts.update(step.inputs)
            referenced_artifacts.update(step.outputs)

        for artifact_name in self.artifacts:
            if artifact_name not in referenced_artifacts:
                issues.append(f"Orphaned artifact: {artifact_name}")

        return issues

    def _load_lineage(self):
        """Load existing lineage data from disk."""
        try:
            lineage_file = self.lineage_dir / 'current_lineage.json'
            if lineage_file.exists():
                with open(lineage_file) as f:
                    data = json.load(f)

                # Restore artifacts
                for _name, artifact_data in data.get('artifacts', {}).items():
                    artifact = DataArtifact(**artifact_data)
                    self.register_artifact(artifact)

                # Restore steps
                for step_data in data.get('steps', []):
                    step = PipelineStep(**step_data)
                    self.record_step(step)

                logger.info(f"Loaded existing lineage: {len(self.artifacts)} artifacts, {len(self.steps)} steps")

        except Exception as e:
            logger.warning(f"Failed to load existing lineage: {e}")

    def save_lineage(self):
        """Save current lineage to disk."""
        try:
            lineage_data = {
                'saved_at': datetime.now().isoformat(),
                'artifacts': {name: art.to_dict() for name, art in self.artifacts.items()},
                'steps': [step.to_dict() for step in self.steps]
            }

            lineage_file = self.lineage_dir / 'current_lineage.json'
            with open(lineage_file, 'w') as f:
                json.dump(lineage_data, f, indent=2, default=str)

            logger.debug(f"Lineage saved to: {lineage_file}")

        except Exception as e:
            logger.error(f"Failed to save lineage: {e}")

    def get_lineage_stats(self) -> dict[str, Any]:
        """
        Get lineage statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_artifacts': len(self.artifacts),
            'total_steps': len(self.steps),
            'graph_nodes': len(self.lineage_graph.nodes),
            'graph_edges': len(self.lineage_graph.edges),
            'lineage_breaks': len(self.detect_lineage_breaks())
        }

        # Artifact type breakdown
        artifact_types = {}
        for artifact in self.artifacts.values():
            art_type = artifact.artifact_type
            artifact_types[art_type] = artifact_types.get(art_type, 0) + 1
        stats['artifact_types'] = artifact_types

        # Step type breakdown
        step_types = {}
        for step in self.steps:
            step_type = step.step_type
            step_types[step_type] = step_types.get(step_type, 0) + 1
        stats['step_types'] = step_types

        return stats


# Global instance
lineage_tracker = DataLineageTracker()
