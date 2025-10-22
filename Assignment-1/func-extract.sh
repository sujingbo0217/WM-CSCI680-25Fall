#!/bin/bash
python func_extractor.py --repos_dir pretrain-repos --output_dir data/pretrain --min_lines 3 --split_type pretrain &&
python func_extractor.py --repos_dir finetune-repos --output_dir data/finetune --min_lines 3 --split_type finetune
