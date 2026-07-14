# RL Exam: DreamerV4 Interactive Inference

This repository contains my final project and presentation for the Reinforcement Learning exam. The focus of this project is to run, analyze, and demonstrate **DreamerV4**, particularly showcasing its robust interactive inference capabilities and world model dynamics.

## Repository Structure

- `dreamer4/`: The main codebase for the DreamerV4 implementation and inference pipeline.
  - `generate_all_seed_shards.py`: A robust script to generate starting state seeds (shards) across all 234 tasks required for inference, featuring placeholder fallbacks when `dm_control` environment generation fails.
  - `interactive.py`: The core web-server application that launches an interactive inference demo, allowing you to visualize the agent's imagined environment dynamically in your browser.
- `dreamer4_tutorial.ipynb`: A Jupyter Notebook tutorial stepping through the key concepts and usage of the DreamerV4 models.
- `paper/2509.24527v1.pdf`: The reference paper for DreamerV4 detailing its architecture and contributions.
- `presentation/Deamer4_presentation.odp`: The final presentation slides for the 30-minute exam session.

## Getting Started

### Running the Interactive Web Demo

To properly run the interactive demo without experiencing blank-frame hallucination issues, you first need to generate the initial seed dataset for all tasks.

1. **Generate Seed Shards:**
   This script will iterate through all tasks and generate the required `.pt` files. It gracefully handles missing `dm_control` environments by generating dynamic dummy placeholders.
   ```bash
   cd dreamer4/dreamer4
   python generate_all_seed_shards.py
   ```

2. **Launch the Server: Desktop Only!**
   Start the interactive web demo by pointing it to the generated shards and the pre-trained checkpoints.
   ```bash
      python interactive.py --tokenizer_ckpt ../logs/tokenizer_ckpts/tokenizer.pt --dynamics_ckpt ../logs/dynamics_ckpts/dynamics.pt --data_dir frames128 --frames_dir frames128
   ```

3. **Launch the Server: Laptop Only!**
   Start the interactive web demo by pointing it to the generated shards and the pre-trained checkpoints.
   ```bash
      python interactive.py --tokenizer_ckpt ../logs/tokenizer_ckpts/tokenizer.pt --dynamics_ckpt ../logs/dynamics_ckpts/dynamics.pt --data_dir frames128 --frames_dir frames128 --amp --ctx_window 16
   ```

4. **Open the Demo:**
   Navigate to the URL provided in the terminal to access the DreamerV4 interactive web interface.