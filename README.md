# Transformer Compression

This repository provides tools and scripts for compressing Transformer-based language models (specifically GPT-2) using various quantization and sparsification strategies with optional error correction and update tasks.

## Features

- Top-K and Importance-based compression strategies
- Optional error correction (EF) with mirror descent or gradient descent update tasks
- Support for multiple restarts and reproducible experiments via seeding
- Integration with Weights & Biases (W&B) for experiment tracking and visualization
- Utilities for training, evaluation, and plotting results

## Repository Structure

```
compress_config.py     # Defines CompressionConfig and experiment setup
compressors.py         # Implementation of compression strategies and compressors
descent.py             # Update and optimization routines (mirror descent, gradient descent)
experiment.py          # Orchestrates full training and evaluation loops
logger.py              # TrainerLogger for logging metrics and saving CSV/plots
main.py                # Entry point: dataset loading, config creation, experiment execution
train.py               # Training and validation functions
utils.py               # Dataset loading, seeding, device management, plotting helpers
requirements.txt       # Python dependencies
plots.ipynb            # Jupyter notebook for interactive plotting and analysis
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/transformer-compression.git
   cd transformer-compression
   ```
2. Create a Python virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your W&B API key in a `.env` file at the project root:
   ```dotenv
   WANDB_API_KEY=your_api_key
   ```

## Usage

Run experiments with main.py. You can adjust compression settings in `main.py` via `param_usage`, `num_epochs`, `num_restarts`, and the list of `CompressionConfig` objects.

Example:
```bash
python main.py
```

All experiment logs, metrics, and artifacts are stored in W&B and local `logs/` and `wandb/` directories.

## Custom Configuration

Edit `main.py` or supply your own list of `CompressionConfig` instances to:
- Change compression strategy (`TopK`, `ImpK`, `SCAM`, etc.)
- Enable error correction (set `error_correction='EF'`)
- Specify update tasks (`mirror_descent_full`, `gradient_descent_full`)
- Tune hyperparameters (`lr`, `eta`, `num_steps`)

## Visualization

- Use `plots.ipynb` for interactive analysis of saved CSV results.
- The `plot_and_save_results` utility in `utils.py` generates static plots and saves them to disk.

## License

This project is released under the MIT License.

## Acknowledgements

- Built with PyTorch and Hugging Face Transformers
- Experiment tracking via Weights & Biases