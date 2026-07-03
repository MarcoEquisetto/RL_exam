# Architecture Documentation: DreamerV4 PyTorch Implementation

This document provides a detailed breakdown of the `dreamer4` PyTorch codebase to help you map the concepts from the "Training Agents Inside of Scalable World Models" paper directly to the code. You can use this as a script/reference for the code-analysis portion of your presentation.

## Overview

The DreamerV4 architecture replaces traditional Recurrent State Space Models (RSSMs) with a highly scalable **Block-Causal Transformer**. This enables the model to efficiently compress temporal dependencies and learn dynamics through a shortcut forcing objective.

The implementation is primarily split into three core components, found in `model.py`:
1.  **Encoder**: Compresses raw image patches into a sequence of low-dimensional latent vectors.
2.  **Decoder**: Reconstructs image patches from the latent sequence.
3.  **Dynamics**: The actual "World Model" predicting future latent states given past states and actions.

---

## 1. The Tokenizer (Encoder & Decoder)

In the paper, the Tokenizer is responsible for translating the high-dimensional visual world into compact discrete representations, and vice-versa. 

### `Encoder` Class (`model.py`)
-   **Role**: Converts (Batch, Time, Patches, PatchDim) into a sequence of latent vectors `z`.
-   **Key Mechanism**: It uses a `BlockCausalTransformer` that alternates between **spatial** attention (within a single frame) and **temporal** attention (across frames).
-   **Masked Autoencoder (MAE)**: The encoder uses `MAEReplacer` to mask out random patches (up to 90%). The decoder must learn to reconstruct these masked patches, forcing the latent `z` to capture robust semantic meaning rather than just compressing pixels.
-   **Output**: The final latent vector `z` is squeezed through a `tanh` bottleneck to bind the values strictly between `[-1, 1]`.

### `Decoder` Class (`model.py`)
-   **Role**: Takes the bottleneck latents `z` and projects them back to image patches.
-   **Key Mechanism**: Uses learned queries (`patch_queries`) alongside the latent sequence in a `BlockCausalTransformer`. The output goes through a `sigmoid` to yield pixel values in `[0, 1]`.

---

## 2. Action Encoding

Before the agent's actions can interact with the Dynamics model, they must be projected into the same dimension (`d_model`) as the latent states.

### `ActionEncoder` Class (`model.py`)
-   **Role**: Embeds continuous actions (from `[-1, 1]`) into transformer tokens.
-   **Mechanism**: A 2-layer MLP (`fc1`, `fc2`) with `SiLU` activation. It adds a learnable `base` parameter so that even if the action vector is zero (or missing during pre-training), the token still carries meaning.

---

## 3. The Dynamics Model (The Simulator)

This is the core of the World Model. It takes past observations and actions and predicts what the world will look like next.

### `Dynamics` Class (`model.py`)
-   **Role**: The causal transformer that learns environment transition dynamics $p(z_{t+1} | z_{\leq t}, a_{\leq t})$.
-   **Input Tokens**: 
    -   `action_tokens`: The action taken.
    -   `spatial_tokens`: The encoded state of the world.
    -   `reg`: Register tokens (acting as extra memory for the transformer).
    -   `step_tok` & `sig_tok`: Used for the "Shortcut Forcing" objective, dictating how many steps into the future the model is predicting.
-   **Attention Modality**: It uses `SpaceSelfAttentionModality` with `mode="wm_agent_isolated"`. This is crucial because it ensures the world model does not "cheat" by looking at future agent tokens, keeping the environment simulation independent of the actor-critic network during training.

### Shortcut Forcing (`sample_one_timestep_packed` in `interactive.py`)
Instead of predicting just $t+1$ and rolling out sequentially, the paper introduces a "Shortcut" schedule. The model can denoise the representation of step $t+k$, allowing for much faster imagined rollouts.
In `interactive.py`, the `sample_one_timestep_packed` function loops over Euler steps `K` (based on the `tau_schedule`) to smoothly denoise a random Gaussian vector into the predicted latent state `z_next`.

---

## Conclusion for Presentation
When presenting the code, emphasize **`BlockCausalTransformer`** and **`sample_one_timestep_packed`**. These are the elements that make DreamerV4 "scalable" compared to previous architectures:
1.  **Transformer vs RNN**: Replacing the recurrent RSSM with causal transformers allows massive parallelization.
2.  **Shortcut Forcing**: Denoising future states allows the agent to imagine long-horizon futures without sequentially stepping through every single frame.
