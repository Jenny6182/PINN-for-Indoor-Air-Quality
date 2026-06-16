"""
Pipeline factory for creating pipeline instances by model type.
"""

from pathlib import Path


def get_pipeline(model_type: str, config_path: str = None, run_dir: Path = None):
    """
    Factory function to get the right pipeline.
    
    Parameters
    ----------
    model_type : str
        One of: 'simple', 'varying', 'raapinn', 'one_raapinn'
    config_path : str
        Path to config YAML file
    run_dir : Path
        Output directory
        
    Returns
    -------
    Pipeline instance
    """
    from experiment.pipelines.simple import SimplePipeline
    from experiment.pipelines.varying import VaryingPipeline
    from experiment.pipelines.raapinn import RAAPINNPipeline
    from experiment.pipelines.one_raapinn import OneRAAPINNPipeline
    
    pipelines = {
        'simple': SimplePipeline,
        'varying': VaryingPipeline,
        'raapinn': RAAPINNPipeline,
        'one_raapinn': OneRAAPINNPipeline,
    }
    
    if model_type not in pipelines:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be one of: {list(pipelines.keys())}"
        )
    
    pipeline_class = pipelines[model_type]
    return pipeline_class(config_path=config_path, run_dir=run_dir)
