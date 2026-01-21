# Research Papers 📚

A curated collection of cutting-edge AI research papers focused on **parameter-efficient fine-tuning**, **large language models**, **efficient inference**, and **recursive reasoning architectures**.

---

## Papers in This Collection

### 1. LoRA Without Regret (Thinking Machines Lab)
**File:** `thinkingmachines-ai-blog-lora-.pdf`  
**Authors:** John Schulman et al. (Thinking Machines Lab)  
**Published:** September 2025

#### 📝 Summary
A groundbreaking study demonstrating that **LoRA (Low-Rank Adaptation)** can achieve performance comparable to full fine-tuning when configured correctly, using approximately **67% less computational resources**.

#### 🔑 Key Findings
- LoRA performs best when applied to **all weight matrices** (including MLP/MoE layers), not just attention layers
- The optimal learning rate for LoRA is consistently **~10x higher** than for full fine-tuning
- LoRA is particularly effective for **reinforcement learning tasks** where it can match full fine-tuning even with very low adapter ranks
- Establishes a "low-regret regime" where LoRA matches full fine-tuning performance in typical post-training scenarios

#### 💡 Core Insight
LoRA's effectiveness comes from its ability to rapidly adapt models to new task formats while preserving the base model's underlying knowledge. The key is proper configuration—not the technique itself being limited.

---

### 2. BitNet a4.8: 4-bit Activations for 1-bit LLMs
**File:** `2411.04965v1.pdf`  
**arXiv:** [2411.04965](https://arxiv.org/abs/2411.04965)  
**Published:** November 2024

#### 📝 Summary
An advancement on the 1-bit LLM architecture (BitNet b1.58), introducing **4-bit activations** for even more efficient inference. Uses a hybrid quantization and sparsification strategy to handle outlier channels.

#### 🔑 Key Findings
- Employs **4-bit activations** for inputs to attention and FFN layers
- Sparsifies intermediate states followed with 8-bit quantization
- Achieves performance **comparable to BitNet b1.58** with equivalent training costs
- Activates only **55% of parameters** during inference
- Supports **3-bit KV cache** for further memory efficiency
- Enables use of optimized INT4/FP4 kernels for faster inference

#### 💡 Core Insight
Extreme quantization (1-bit weights + 4-bit activations) is viable for LLMs when combined with intelligent sparsification. The key innovation is handling outlier activation channels through hybrid strategies rather than uniform quantization.

---

### 3. Tina: Tiny Reasoning Models via LoRA
**File:** `2504.15777v1.pdf`  
**arXiv:** [2504.15777](https://arxiv.org/abs/2504.15777)  
**Published:** April 2025

#### 📝 Summary
Demonstrates that **strong reasoning abilities can be achieved extremely cost-effectively** by applying LoRA during reinforcement learning on tiny (1.5B parameter) base models. Achieves SOTA-competitive reasoning at a fraction of the cost.

#### 🔑 Key Findings
- Achieves **>20% reasoning performance increase** on AIME24 benchmark
- **43.33% Pass@1 accuracy** on AIME24 with only **$9 USD** post-training cost
- Represents an estimated **260x cost reduction** compared to existing SOTA methods
- Uses parameter-efficient LoRA updates during RL, not full fine-tuning
- Effectiveness validated across multiple reasoning datasets

#### 💡 Core Insight
LoRA's efficiency during RL stems from its ability to **rapidly adapt the model to the structural format of reasoning** rewarded by RL, while largely preserving the base model's underlying knowledge. You don't need massive compute to create reasoning models—just smart adaptation.

#### 🔗 Resources
- Fully open-source: code, training logs, model weights & checkpoints

---

### 4. Less is More: Recursive Reasoning with Tiny Networks (HRM/TRM)
**File:** `2510.04871v1.pdf`  
**arXiv:** [2510.04871](https://arxiv.org/abs/2510.04871)  
**Published:** October 2025

#### 📝 Summary
Introduces **Hierarchical Reasoning Model (HRM)** and **Tiny Recursive Model (TRM)**—novel approaches using small neural networks that reason through **recursion**. Beats LLMs on hard puzzle tasks with tiny models trained on minimal data.

#### 🔑 Key Findings
- **HRM**: Two small networks recursing at different frequencies (biologically inspired)
- **TRM**: Even simpler—achieves higher generalization with a **single 2-layer network**
- With only **7M parameters**, TRM achieves:
  - **45% test-accuracy on ARC-AGI-1**
  - **8% on ARC-AGI-2**
- Outperforms most LLMs (DeepSeek R1, o3-mini, Gemini 2.5 Pro) with **<0.01% of the parameters**
- Trained on only ~1000 examples

#### 💡 Core Insight
**Recursion > Scale** for certain reasoning tasks. Instead of making models bigger, make them think iteratively. The recursive structure allows tiny networks to solve problems that stump massive LLMs by refining solutions through multiple passes.

---

### 5. LFM2 Technical Report (Liquid Foundation Models 2)
**File:** `2511.23404v1.pdf`  
**arXiv:** [2511.23404](https://arxiv.org/abs/2511.23404)  
**Published:** November 2025  
**Organization:** Liquid AI

#### 📝 Summary
A comprehensive technical report on **Liquid Foundation Models 2 (LFM2)**—a family of models designed for efficient on-device deployment. Uses hardware-in-the-loop architecture search to create compact hybrid backbones optimized for edge devices.

#### 🔑 Key Findings
- **Model sizes:** 350M, 700M, 1.2B, 2.6B (dense) + 8.3B MoE (1.5B active)
- **32K context length** across all models
- **2x faster prefill and decode** on CPUs compared to similar-sized models
- Architecture: Hybrid of **gated short convolutions + grouped query attention**
- Training: 10-12T tokens with curriculum learning and knowledge distillation
- **LFM2-2.6B** achieves 79.56% on IFEval and 82.41% on GSM8K

#### 🧩 Model Variants
| Variant | Purpose |
|---------|---------|
| **LFM2-VL** | Vision-language tasks with tunable accuracy-latency tradeoffs |
| **LFM2-Audio** | Real-time speech-to-speech (competitive with 3x larger models) |
| **LFM2-ColBERT** | Low-latency multi-language retrieval encoder |

#### 💡 Core Insight
Liquid AI's approach combines **architecture search under real hardware constraints** with novel training techniques (tempered knowledge distillation, curriculum learning, preference optimization + model merging). The result is models specifically optimized for edge deployment, not just scaled-down cloud models.

#### 🔗 Resources
- Open weights + deployment packages for ExecuTorch, llama.cpp, vLLM

---

### 6. ArcAligner: Adaptive Recursive Aligner for Compressed Context in RAG
**File:** `2601.05038v1.pdf`  
**arXiv:** [2601.05038](https://arxiv.org/abs/2601.05038)  
**Published:** January 2026

#### 📝 Summary
Addresses the **compression-comprehension tradeoff in RAG**: more context compression = harder for LLMs to understand. Introduces **ArcAligner**, a lightweight module that helps models better utilize highly compressed context representations.

#### 🔑 Key Findings
- Proposes **Adaptive recursive context Aligner (ArcAligner)**
- Integrates directly into language model layers
- Uses **adaptive gating**—only adds extra processing when information is complex
- Beats compression baselines consistently, especially on:
  - **Multi-hop reasoning** tasks
  - **Long-tail** knowledge settings
- Maintains speed while improving compressed context understanding

#### 💡 Core Insight
The problem with aggressive RAG compression isn't the compression itself—it's the model's inability to decode compressed representations. ArcAligner solves this by teaching models to **adaptively align** compressed embeddings back to usable representations, with compute scaling based on complexity.

#### 🔗 Resources
- Source code publicly available

---

### 7. EMNLP 2025 Findings Paper
**File:** `2025.findings-emnlp.321.pdf`  
**Venue:** EMNLP 2025 Findings Track

#### 📝 Summary
A paper from the **EMNLP 2025 Findings** track—one of the premier venues for empirical NLP research. The Findings track publishes solid research that may not fit the main conference but represents valuable contributions to the field.

*Note: Specific abstract pending—please refer to ACL Anthology for full details.*

---

## 🔗 How These Papers Connect

```
┌─────────────────────────────────────────────────────────────────┐
│                    EFFICIENT AI STACK                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  BitNet a4.8 │ +  │    LFM2      │ =  │ Ultra-compact│      │
│  │  (1-bit LLM) │    │ (Liquid AI)  │    │ Edge Models  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────┬───────┘                   │               │
│                     ▼                           │               │
│           ┌──────────────┐                      │               │
│           │   HRM / TRM  │◄─────────────────────┘               │
│           │  (Recursive  │                                      │
│           │   Reasoning) │                                      │
│           └──────────────┘                                      │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │ Tina (LoRA   │◄───────►│ LoRA Without │                     │
│  │  + RL)       │         │    Regret    │                     │
│  └──────────────┘         └──────────────┘                     │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  ArcAligner  │  ← Compressed RAG for knowledge augmentation │
│  │  (RAG)       │                                              │
│  └──────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Cutting-Edge Researchers to Follow

### John Schulman
**Chief Scientist, Thinking Machines Lab**  
🔗 [Personal Website](https://joschu.net) | [Google Scholar](https://scholar.google.com/citations?user=jYVDNP4AAAAJ)

- PhD from UC Berkeley (Robotics & RL under Pieter Abbeel)
- Co-founder of OpenAI, led the ChatGPT reinforcement learning team
- Pioneered foundational RL algorithms: **PPO (Proximal Policy Optimization)**, **TRPO**
- Research focus: Reinforcement learning, language model alignment, model transparency

### Mira Murati
**Founder & CEO, Thinking Machines Lab**  
🔗 [LinkedIn](https://www.linkedin.com/in/mikimurati/) | [Twitter](https://twitter.com/maborovska)

- Former CTO of OpenAI
- Founded Thinking Machines Lab in February 2025
- Expert in AI systems development and scaling

### Ramin Hasani & Mathias Lechner
**Co-founders, Liquid AI**  
🔗 [Liquid AI](https://www.liquid.ai/)

- Pioneers of Liquid Neural Networks (Liquid Time-constant Networks)
- Research on biologically-inspired adaptive computation
- Leading edge-optimized AI deployment

### Pieter Abbeel
**Professor, UC Berkeley | Co-founder, Covariant**  
🔗 [Personal Website](https://people.eecs.berkeley.edu/~pabbeel/) | [Google Scholar](https://scholar.google.com/citations?user=vtwH6GkAAAAJ)

- Leading expert in robotics and reinforcement learning
- John Schulman's PhD advisor
- Research focus: Robot learning, deep RL, imitation learning

---

## Transformer Architecture Pioneers

### Ashish Vaswani
**Co-author of "Attention Is All You Need"**  
🔗 [Google Scholar](https://scholar.google.com/citations?user=oR9sCGYAAAAJ)

- Original architect of the transformer model
- Founding research at Google Brain

### Noam Shazeer
**Co-founder, Character.AI**  
🔗 [Google Scholar](https://scholar.google.com/citations?user=jVT21XAAAAAJ)

- Co-author of the original transformer paper
- Pioneered many innovations in LLM architecture

### Llion Jones
**Researcher, Sakana AI**  
🔗 [Twitter](https://twitter.com/lllooonnnzzz)

- Co-author of original transformer paper
- Currently researching biologically-inspired architectures (Continuous Thought Machine)
- Questioning whether current transformer scaling is the right path

---

## Leading AI Research Labs & Researchers

### Sebastian Raschka
**AI Educator & Researcher**  
🔗 [Website](https://sebastianraschka.com) | [GitHub](https://github.com/rasbt)

- Prolific educator on LLMs and deep learning
- Author of comprehensive LLM tutorials and research

### Andrej Karpathy
**Founder, Eureka Labs**  
🔗 [Personal Website](https://karpathy.ai) | [YouTube](https://www.youtube.com/@AndrejKarpathy)

- Former Director of AI at Tesla, founding member of OpenAI
- Known for educational content on deep learning

---

## 🏢 Key Research Organizations

| Organization | Focus | Notable Models/Research |
|-------------|-------|-------------------------|
| **OpenAI** | Frontier AI research | GPT-4, GPT-5, o1/o3 reasoning |
| **Google DeepMind** | Multimodal AI, scaling | Gemini series |
| **Anthropic** | AI safety & alignment | Claude series |
| **Meta AI** | Open research | LLaMA, OPT |
| **Mistral AI** | Efficient LLMs | Mistral 7B, Mixtral |
| **Thinking Machines Lab** | Post-training research | LoRA Without Regret |
| **Liquid AI** | Edge-optimized models | LFM2 family |
| **Sakana AI** | Bio-inspired architectures | CTM research |
| **Microsoft Research** | Quantization | BitNet series |

---

## 📖 How to Use This Repository

1. Each PDF file contains the full paper for study
2. Read this README for quick summaries and core insights
3. Follow the "How These Papers Connect" diagram to understand relationships
4. Check arXiv links for the latest versions of papers
5. Follow the researchers on linked platforms for updates

---

*Last updated: January 2026*
