class FeatureExtractor:
    """
    Extracts decision features for adaptive checkpoint placement.
    Combines structural metrics + runtime state.
    """

    def __init__(self, structural_metrics):
        self.structural_metrics = structural_metrics

    def extract(
        self,
        work_since_last_checkpoint,
        failure_rate,
        avg_block_cost,
        execution_variance=0.0
    ):
        total_blocks = self.structural_metrics["total_basic_blocks"]
        loop_count = self.structural_metrics["loop_count"]
        branch_count = self.structural_metrics["branch_count"]
        cyclomatic_complexity = self.structural_metrics["cyclomatic_complexity"]

        loop_density = loop_count / total_blocks if total_blocks else 0
        branch_density = branch_count / total_blocks if total_blocks else 0

        features = {
            "work_since_last_checkpoint": work_since_last_checkpoint,
            "failure_rate": failure_rate,
            "avg_block_cost": avg_block_cost,
            "execution_variance": execution_variance,
            "loop_density": loop_density,
            "branch_density": branch_density,
            "cyclomatic_complexity": cyclomatic_complexity
        }

        return features