<div align="center">
    <img alt="OpenRLHF logo" src="./docs/logo.png" style="height: 160px;" />
</div>
<div align="center">
<p align="center">
      <a href="https://github.com/openllmai/OpenRLHF/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/openllmai/OpenRLHF" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/openllmai/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/openllmai/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/openllmai/OpenRLHF?color=0088ff" />
      <a href="https://github.com/openllmai/OpenRLHF/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/openllmai/OpenRLHF?color=ccf" />
      </a>
      <br>
      <em>Open-source / Comprehensive / Lightweight / Easy-to-use</em>
    </p>
</p>
</div>

<hr>

<span>[ English | <a href="README_zh.md">中文</a> ]</span>

OpenRLHF is a high-performance RLHF framework built on Ray, DeepSpeed and HF Transformers:

- **Simple and easy to use**: OpenRLHF is one of the simplest high-performance RLHF libraries currently available, and compatible with Huggingface models and datasets.
- **High performance**: RLHF training spends 80% of the time on the sample generation stage. Thanks to the ability to use a large inference batch size with Ray and Adam Offload (Pinned Memory) and vLLM generation acceleration, the performance of OpenRLHF 2x+ that of Optimized DeepSpeedChat with Hybrid Engine.
- **Distributed RLHF**:  OpenRLHF distribute the Actor, Reward, Reference, and Critic models onto separate GPUs using Ray, while placing the Adam optimizer on the CPU. This enables full-scale fine-tuning of 70B+ models with multiple A100 80G GPUs and vLLM and 7B models across multiple 24GB RTX 4090 GPUs.
- **PPO Implementation Optimization**: We integrated the implementation tricks for PPO to improve the training stability, referencing [Zhihu](https://zhuanlan.zhihu.com/p/622134699) and the [Notion blog](https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f).

More details are in [Technical Report](https://arxiv.org/abs/2405.11143) | [Documents](https://openrlhf.readthedocs.io/)


## Features

- Distributed [PPO based on Ray](./examples/scripts/train_ppo_llama_ray.sh). 
- Support full RLHF fine-tuning of models with [over 70 billion parameters](./examples/scripts/train_ppo_llama_ray_70b.sh).
- Support vLLM generation acceleration in RLHF (--vllm_num_engines).
- Support multiple reward models (--reward_pretrain model1,model2...).
- Support [DPO (direct-preference-optimization)/IPO/cDPO](./examples/scripts/train_dpo_llama.sh).
- Support [Kahneman-Tversky optimization (KTO)](./examples/scripts/train_kto_llama.sh).
- Support [Rejection Sampling](./examples/scripts/train_rejection_sampling_llama.sh).
- Support [Iterative DPO](./examples/scripts/train_iterative_dpo_llama.sh) (https://github.com/RLHFlow/Online-RLHF).
- Support [Conditional SFT](./examples/scripts/train_conditional_llama.sh) (https://arxiv.org/abs/2308.12050).
- Support [Knowledge Distillation](./examples/scripts/train_knowledge_distillation.sh) (https://github.com/microsoft/LMOps/tree/main/minillm).
- Support [MoE](./examples/test_scripts/train_sft_mixtral_lora.sh) (--aux_loss_coef)
- Support Wandb log (--wandb).
- Support FlashAttention2 (--flash_attn).
- Support QLoRA (--load_in_4bit), [LoRA (--lora_rank, --target_modules)](./examples/scripts/train_sft_mixtral_lora.sh).
- Support HuggingFace `tokenizer.apply_chat_template` in datasets (--apply_chat_template and --input_key).
- Multi-nodes [training scripts](./examples/scripts/train_llama_slurm.sh) for Slurm.

### PPO Support Matrix

| Feature | OpenRLHF | DSChat | CAIChat | TRL |
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:|
| 70B+ Full Tuning with 16 A100-80GB      | ✅ | ❌ | ❌ | ❌ |
| 7B Full Tuning with 4 RTX4090 | ✅      |    ❌ | ❌ | ❌ |
| 34B DPO Full Tuning with 8 A100-80GB | ✅      |    ❌ | ❌ | ❌ |  
| Inference Engine in PPO | ✅      |    ✅ | ❌ | ❌ |  
| PPO Implementation Tricks | ✅      |    ❌ | ❌ | ✅ |
| Support QLoRA | ✅      |    ❌ | ❌ | ✅ | 
| Support Mixtral 8*7b | ✅      |    ❌ | ❌ | ❌ |  
| Support Unmerged Actor-Critic | ✅     |   ✅ | ✅ | ❌ | 
| Support Multiple Reward Models | ✅      |    ❌ | ❌ | ❌ |   
| Support Huggingface Models | ✅      |    ✅ | ✅ | ✅ | 
| Easy-to-use | ✅      |   ❌ (HybridEngine bugs) | ✅ | ✅ | 


## Quick Start

### Installation

To use OpenRLHF, first ``git clone`` it and launch the docker container (**Recommended**):

```bash
git clone https://github.com/openllmai/OpenRLHF.git

# If you need to use vLLM, please build a Docker image to avoid dependency issues (Optional)
docker build -t nvcr.io/nvidia/pytorch:24.02-py3 ./OpenRLHF/dockerfile

# Launch the docker container
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD/OpenRLHF:/openrlhf nvcr.io/nvidia/pytorch:24.02-py3 bash
```

> [!NOTE]
>We provided the [One-Click Installation Script of Nvidia-Docker](./examples/scripts/nvidia_docker_install.sh)

Then ``pip install`` openrlhf inside the docker container

```bash
cd /openrlhf
pip install --user .

cd examples
```

### Prepare Datasets
OpenRLHF provides multiple data processing methods in our dataset classes.
Such as in the [Prompt Dataset](https://github.com/OpenLLMAI/OpenRLHF/blob/7e436a673b9603847429971290cfd46029c4b52b/openrlhf/datasets/prompts_dataset.py#L6):

```python
def preprocess_data(data, input_template=None, input_key=None, apply_chat_template=None) -> str:
  # custom dataset
  if input_key:
      if apply_chat_template:
        prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
        input_template = None
      else:
        prompt = data[input_key]
  else:
      # Open-Orca/OpenOrca
      if exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
        prompt = data["system_prompt"] + " " + data["question"]
      .....

  # input template
  if input_template:
      prompt = input_template.format(prompt)
  return prompt
```

- We can use `--input_key` to specify the `JSON key name` of the input datasets `--prompt_data {name or path}` (PPO) or `--dataset {name or path}`, and use `--apply_chat_template` to utilize the `chat_template` from the [Huggingface Tokenizer](https://huggingface.co/docs/transformers/main/en/chat_templating).
- If you don't want to use `--apply_chat_template`, you can use `--input_template` instead, or preprocess the datasets offline in advance.
- OpenRLHF also support mixing multiple datasets using `--prompt_data_probs 0.1,0.4,0.5` (PPO) or `--dataset_probs 0.1,0.4,0.5`.

Chat Templating

```python
dataset = [{"input_key": [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]}]

tokenizer.apply_chat_template(dataset[0]["input_key"], tokenize=False)

"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
```
> [!NOTE]
> The ``JSON key`` options depends on the specific datasets. See [Reward Dataset](https://github.com/OpenLLMAI/OpenRLHF/blob/7e436a673b9603847429971290cfd46029c4b52b/openrlhf/datasets/reward_dataset.py#L10) and [SFT Dataset](https://github.com/OpenLLMAI/OpenRLHF/blob/7e436a673b9603847429971290cfd46029c4b52b/openrlhf/datasets/sft_dataset.py#L9)

### Supervised Fine-tuning

OpenRLHF's model checkpoint is fully compatible with HuggingFace models. You can specify the model name or path using `--pretrain  {name or path}`, `--reward_pretrain  {name or path}` and `--critic_pretrain  {name or path}`. We have provided some pre-trained checkpoints and datasets on [HuggingFace OpenLLMAI](https://huggingface.co/OpenLLMAI).

Then you can use the startup scripts we provide in the [examples/scripts](./examples/scripts/) directory, or start the training using the following commands.

```bash 
deepspeed ./train_sft.py \
   --max_len 2048 \
   --dataset Open-Orca/OpenOrca \
   --dataset_probs 1.0 \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain meta-llama/Llama-2-7b-hf \
   --save_path ./checkpoint/llama2-7b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --use_wandb {wandb_token}

# Customization of chat_template is supported.
# --apply_chat_template 
# --input_key {JSON Key}
# --tokenizer_chat_template {HF Chat Template}

# Can also be used for continued pre-training
# --pretrain_mode
```

### Reward Model Training
```bash
deepspeed ./train_rm.py \
   --save_path ./checkpoint/llama3-8b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset OpenLLMAI/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb {wandb_token}
```

### PPO without Ray

```bash
deepspeed ./train_ppo.py \
  --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
  --reward_pretrain OpenLLMAI/Llama-3-8b-rm-mixture \
  --save_path ./checkpoint/llama-3-8b-rlhf \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 1024 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data OpenLLMAI/prompt-collection-v0.1 \
  --input_key context_messages \
  --apply_chat_template \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb {wandb_token}
```

### PPO with Ray and vLLM

To improve RLHF training speed or support 70B models, we can use the PPO with Ray and vLLM acceleration

```bash
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/openrlhf", "pip": "/openrlhf/requirements.txt"}' \
  -- python3 examples/train_ppo_ray.py \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 2 \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node 2 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 2 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 2 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 2 \
  --colocate_critic_reward \
  --colocate_actor_ref \
  --ref_reward_offload \
  --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
  --reward_pretrain OpenLLMAI/Llama-3-8b-rm-mixture \
  --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
  --micro_train_batch_size 8 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 16 \
  --rollout_batch_size 1024 \
  --max_samples 100000 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data OpenLLMAI/prompt-collection-v0.1 \
  --input_key context_messages \
  --apply_chat_template \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb {wandb_token}
```

> [!NOTE]
> We recommend using vLLM 0.4.2, as versions 0.4.3+ currently only support weight synchronization (DeepSpeed => vLLM) via GLOO (--vllm_sync_backend gloo).

The launch scripts and docs for all supported algorithms are in [example/scripts](./examples/scripts/) and [Documents - Usage](https://openrlhf.readthedocs.io/en/latest/usage.html)

## Performance

We optimized DSChat's performance to the greatest extent possible by employing techniques such as enabling Adam offload, along with reward model (RM) and reference model (Ref) offload to increase the micro-batch size during the inference stage and avoid out-of-memory issues. We even fixed some bugs in DSChat to enable the Hybrid Engine (HE) for LLaMA2. The average time (seconds) it took to train 1024 prompts with 1 PPO epoch using the Optimized DSChat and OpenRLHF:

| **Size** | **NVIDIA A800-80GB GPUs** | **Optimized DSChat (with  Hybrid Engine)** | **OpenRLHF** | **Speedup** |
| :---: | :---: | :---: | :---: | :---: |
| 7B | 16 | 855.09 | 471.11 | 1.82x |
| 13B | 32 | 1528.93 | 608.93 | 2.5x |
| 34B | 32 | 3634.98 | 1526.4 | 2.4x |
| 70B | 32 | 10407.0 | 4488.53 | 2.3x |

### Performance Tuning Guide

To achieve optimal performance, we recommend allocating more nodes to the vLLM Engine. For example, for a 70B model with 32 A100 GPUs, it is advised to allocate more than 16 A100 GPUs to the vLLM Engine, 8 GPUs to the Actor model, and the remaining 8 GPUs to the Critic model. Additionally, enable the `--colocate_critic_reward`, `--colocate_actor_ref`, and `--ref_reward_offload` options to merge nodes. Finally, you should increase the micro-batch-size (and minimize the TP size of vLLM engine) as much as possible while avoiding OOM (Out Of Memory) issues, especially during the generation phase of PPO. Enable `enable_prefix_caching` in vLLM generation when `n_samples_per_prompt > 1`.


## Join Us

**How to Join?**

1. Email us at xianyuai@openllmai.top(open-source community email) or janhu9527@gmail.com (personal email of PIC). Please include the following details:
   - Your name
   - Your GitHub username
   - Your areas of interest
   - Your skills and experience related to NLP and/or AI
1. You can also join us through the official GitHub [OpenRLHF ↗](https://github.com/openllmai/OpenRLHF) project page. Just create an issue about your interest to contribute and we will get back to you.

**What can you do?**

1. Join the team and participate in the development of the OpenRLHF project.
1. Contribute to the project by submitting pull requests.
1. Help improve documentation, fix bugs, or create new features.
1. Share the project and help us grow the community.

## Sponsor Us

Your sponsorship can help us maintain and improve OpenRLHF. If you find this project useful, please consider sponsoring us. You can sponsor us on [Open Collective ↗](https://opencollective.com/openllmai).

## Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=openllmai/OpenRLHF&type=Date)](https://star-history.com/#openllmai/OpenRLHF&Date)

## Contributors

A big thank you to all our contributors! If you want to contribute, feel free to make a pull request or create an issue.

<a href="https://github.com/openllmai/OpenRLHF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openllmai/OpenRLHF" />
</a>

## References & Acknowledgements

We would like to express our gratitude to the following projects and organizations for their contributions to the field of AI and NLP:

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA2 ↗](https://ai.meta.com/llama/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)

Our project would also like to thank [ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) and [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat). In the early stages of the project, we referred to their code design.


## Citation
```
@article{hu2024openrlhf,
  title={OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
  author={Jian Hu and Xibin Wu and Weixun Wang and Xianyu and Dehao Zhang and Yu Cao},
  journal={arXiv preprint arXiv:2405.11143},
  year={2024}
}
```

______________________________________________________________________

*OpenRLHF © 2024 OpenLLMAI. All Rights Reserved.*
