# Changelog

All notable changes to the EpiBERT project are documented in this file.

## [1.1.0] - 2024-07-30

### Breaking Changes
- Renamed model classes to avoid registration conflicts:
  - `epibert` → `epibert_atac_pretrain` in `src/models/epibert_atac_pretrain.py`
  - `epibert` → `epibert_rampage_finetune` in `src/models/epibert_rampage_finetune.py`
- Updated training scripts to use new model class names

### Added/Fixed/Changed
- GPU memory management and mixed precision support in notebooks
- Enhanced error handling and logging in example notebooks
- Installation instructions for system dependencies
- Keras serialization conflicts preventing simultaneous model imports
- Model instantiation calls in training scripts
- Import paths and error handling in notebooks
- Reorganized `requirements.txt` with better documentation
- Improved notebook documentation and structure


## [1.0.0] - Original Release
- Initial EpiBERT implementation
- ATAC-seq pre-training and RAMPAGE fine-tuning
- TPU training support
- Basic example notebooks 