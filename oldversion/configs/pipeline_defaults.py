"""
Single place to set the default Gymnasium env for pipeline / training scripts.

Change ``DEFAULT_PIPELINE_ENV_ID`` here instead of typing long ``--env`` strings.
CLI ``--env`` still overrides this default when supported.
"""

# Examples (uncomment one):
# DEFAULT_PIPELINE_ENV_ID = "MiniGrid-Empty-5x5-v0"
# DEFAULT_PIPELINE_ENV_ID = "CausalContrastive-HiddenFork-15x15-Clean-v0"
DEFAULT_PIPELINE_ENV_ID = "CausalContrastive-HiddenFork-15x15-v0"  # random U each episode
