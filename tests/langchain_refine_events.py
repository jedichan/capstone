"""
Convenience entry to run refinement locally.

From repo root (adjust paths to your pipeline outputs):
  python tests/langchain_refine_events.py \\
    --events pipeline_output/xxx_events.json \\
    --clips pipeline_output/xxx_clips.json

Set OPENAI_API_KEY; you can use a .env in this folder.
"""

from video.factory.coordinator import cli_run_refine_events

if __name__ == "__main__":
    cli_run_refine_events()
