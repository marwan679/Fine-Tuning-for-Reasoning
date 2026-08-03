# Gemma-3-1B-Instruct GRPO Math Reasoning Fine-Tuner

A specialized Google Colab / local notebook implementation showcasing reinforcement learning through **GRPO (Group Relative Policy Optimization)** using [Unsloth](https://github.com/unslothai/unsloth) and Hugging Face's `trl` library[cite: 1]. 

The notebook takes a lightweight **Gemma 3 1B Instruct** model and teaches it structured mathematical reasoning and reasoning trace generation using the **GSM8K** dataset[cite: 1].

---

## Features

* **⚡ Fast Memory-Efficient Training**: Leverages Unsloth optimizations to patch Gemma 3 for faster, low-VRAM training via 16-bit LoRA adapters[cite: 1].
* **🧠 GRPO Reinforcement Learning**: Implements custom multi-objective reward functions to guide the model:
  * **Exact Formatting Rewards**: Penalizes or rewards structural compliance for custom thinking tags (`<start_working_out>` and `<SOLUTION>`)[cite: 1].
  * **Numerical/Answer Accuracy**: Validates final outputs against ground truth answers using direct checks, string stripping, and proportional proximity ratios[cite: 1].
* **🛠️ Model Export Options**: Ready-to-use cells to save model checkpoints locally as LoRA adapters, merged float16 models for vLLM deployment, or quantized GGUF weights for `llama.cpp`[cite: 1].

---

## Requirements

The workflow is designed primarily to run seamlessly in Google Colab with a T4 GPU (or higher)[cite: 1]. Dependencies are automatically managed via `uv` and `pip` directly inside the installation blocks[cite: 1]:
* `unsloth`[cite: 1]
* `vllm`[cite: 1]
* `trl`[cite: 1]
* `torch` & `transformers`[cite: 1]
* `datasets`[cite: 1]

---

## Quickstart Guide

1. Open the provided notebook environment (e.g., Google Colab with a GPU accelerator enabled)[cite: 1].
2. Run the **Installation** block to install Unsloth, vLLM, and core dependencies with optimized package resolutions[cite: 1].
3. Execute the setup cells to load `unsloth/gemma-3-1b-it` and configure the LoRA target layers[cite: 1].
4. Prepare the **GSM8K** dataset and configure the system prompt with structured reasoning tags[cite: 1].
5. Initialize the `GRPOTrainer` with custom reward functions (`match_format_exactly`, `check_answer`, etc.) and launch training[cite: 1].
6. Run the **Inference** section to test your newly trained math reasoning agent[cite: 1].

---

## Saving & Exporting

Once training is complete, you can export your model using the scripts provided at the end of the pipeline[cite: 1]:
* **LoRA Adapters**: `model.save_pretrained("gemma_3_lora")`[cite: 1]
* **Merged 16-bit**: `model.save_pretrained_merged("gemma-3-finetune", tokenizer)`[cite: 1]
* **GGUF Format**: `model.save_pretrained_gguf("gemma_3_finetune", tokenizer, quantization_method = "Q8_0")`[cite: 1]

---

## License
Free open-source license provided via Unsloth[cite: 1]. Refer to individual model cards for Gemma 3 usage terms.
