#!/usr/bin/env python3
import sys
from pipeline.pipeline import SFTLPipeline

if __name__ == "__main__":
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    pipeline = SFTLPipeline(config_path)
    pipeline.run()