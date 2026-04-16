# Stable Language Guidance for Vision-Language-Action Models

<p align="center">
<b>ACL 2026 Main Conference</b>
</p>


<p align="center">
<a href="https://arxiv.org/abs/2601.04052"><img src="https://img.shields.io/badge/arXiv-Paper-red" alt="arXiv"></a>
<a href="https://github.com/Doo-mon/RSS"><img src="https://img.shields.io/badge/Code-GitHub-green" alt="Code"></a>
</p>

<p align="center">
Zhihao Zhan<sup>1</sup>, Yuhao Chen<sup>1</sup>, Jiaying Zhou<sup>1</sup>, Qinhan Lyu<sup>1</sup>, Hao Liu<sup>1</sup>, Keze Wang<sup>1,2,3</sup>, Liang Lin<sup>1,2,3</sup>, Guangrun Wang<sup>*1,2,3</sup>
</p>

<p align="center">
<sup>1</sup>Sun Yat-sen University, <sup>2</sup>Guangdong Key Laboratory of Big Data Analysis and Processing, <sup>3</sup>X-Era AI Lab, <sup>4</sup>Guangdong University of Technology.
</p>

<p align="center">
<sup>*</sup>Corresponding author 
</p>


## ✨ Abstract
![Pipeline](figs/rss-figs.png)

Vision-Language-Action (VLA) models have demonstrated impressive capabilities in generalized robotic control; however, they remain notoriously brittle to linguistic perturbations. We identify a critical ``modality collapse'' phenomenon where strong visual priors overwhelm sparse linguistic signals, causing agents to overfit to specific instruction phrasings while ignoring the underlying semantic intent. To address this, we propose **Residual Semantic Steering (RSS)**, a probabilistic framework that disentangles physical affordance from semantic execution. RSS introduces two theoretical innovations: (1) **Monte Carlo Syntactic Integration**, which approximates the true semantic posterior via dense, LLM-driven distributional expansion, and (2) **Residual Affordance Steering**, a dual-stream decoding mechanism that explicitly isolates the causal influence of language by subtracting the visual affordance prior. Theoretical analysis suggests that RSS effectively maximizes the mutual information between action and intent while suppressing visual distractors. Empirical results across diverse manipulation benchmarks demonstrate that RSS achieves state-of-the-art robustness, maintaining performance even under adversarial linguistic perturbations.





## ⚙️ Setup

### uv
We manage Python dependencies with [uv](https://docs.astral.sh/uv/). If you haven't installed `uv`, please follow [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up.

Run the following to set up the environment:

```bash
git clone --recurse-submodules git@github.com:Doo-mon/RSS.git

# Or if you already cloned the repo:
git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

For more details, refer to the original [openpi repository](https://github.com/Physical-Intelligence/openpi).


## 🚀 Training / Inference / Deployment

### Caption Data Preparation

Refers to ```/examples/libero_shortcut/convert_libero_caption_data_to_lerobot_intern.py```, ```/examples/libero_shortcut/convert_libero_caption_data_to_lerobot_llava.py``` or ```/examples/libero_shortcut/convert_libero_caption_data_to_lerobot_qwen.py```


### Multiple Inference Settings
Refefs to ```/examples/libero_shortcut/main_{*}.py```


### Commands

Refers to `run_train.sh`, `run_server_eval.sh` and `run_local_eval.sh`




## Citation
If you find our work useful, please consider citing:

```bibtex
@article{zhan2026stable,
  title={Stable Language Guidance for Vision-Language-Action Models},
  author={Zhan, Zhihao and Chen, Yuhao and Zhou, Jiaying and Lv, Qinhan and Liu, Hao and Wang, Keze and Lin, Liang and Wang, Guangrun},
  journal={arXiv preprint arXiv:2601.04052},
  year={2026}
}
```

## Acknowledgements

We express our sincere gratitude to the developers of [openpi](https://github.com/Physical-Intelligence/openpi) for open-sourcing their codebase.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
