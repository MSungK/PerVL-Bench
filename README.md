# PerVL-Bench: Benchmarking Multimodal Personalization for Large Vision–Language Models


**Abstract:** In recent years, personalization, which utilizes user-specific data to generate tailored responses, has been increasingly adopted in user-centric domains. However, while Large Language Models (LLMs) are actively researched, the exploration of the personalization capabilities of Large Vision-Language Models (LVLMs) remains limited. To systematically evaluate the personalization ability of LVLMs, we introduce PerVL-Bench, a synthetic benchmark specifically designed for this purpose. PerVL-Bench incorporates user-specific data, including multiple images and long text information, and provides two types of QA pairs. Furthermore, we use PerVL-Bench to comprehensively evaluate the essential capabilities for personalization in current state-of-the-art LVLMs. Through this evaluation, we reveal the limitations of current models in multimodal personalization and provide insights for the development of personalized LVLMs.

---
## Dataset

### Example
![image](assets/Example.png)

### Download
The dataset proposed in this paper can be downloaded [here](https://drive.google.com/file/d/1iITRrs_CICwCiB9p3fEun0xLTMqU0-Ri/view?usp=drive_link).

### File Structure
```plain text
.
├── assets
├── data
│   ├── benchmark
│   ├── database
│   ├── query_images
│   └── vp_images
├── main_results
├── scripts
└── src
```

---
## Code

### Installation
1. Clone this repository:
```shell
git clone https://github.com/MSungK/PerVL-Bench.git
cd PerVL-Bench
```
2. Set up your Python environment:
```shell
conda create -n pervl python=3.10 -y
conda activate pervl
pip install -r requirements.txt
```
3. Set up your API keys:
```shell
export openai_api_key="your_openai_api_key_here"
export genai_api_key="your_genai_api_key_here"
export deepinfra_api_key="your_genai_api_key_here"
```

### Inference

#### Inference via API call with GPT, Gemini, and Gemma series models
This code supports multi-processing to enable faster inference when using API calls. The number of processes can be adjusted via {total_shard_num} in the script.
- Performing **Text-prompt QA** inference:
```shell
sh scripts/text_prompt_qa_API.sh
```

- Performing **Multimodal-prompt QA** inference:
```shell
sh scripts/multimodal_prompt_qa_API.sh
```

When using multi-processing, the following procedure is required to aggregate the results from each process:
```shell
python src/output_merge.py --name {output_directory}
```

#### Inference via HF with Qwen series models.
- Performing **Text-prompt QA** inference:
```shell
sh scripts/text_prompt_qa_HF.sh
```

- Performing **Multimodal-prompt QA** inference:
```shell
sh scripts/multimodal_prompt_qa_HF.sh
```

### Evaluation
The results used in this paper can be found in the main_results directory.

#### Text-prompt QA
```shell
python src/evaluation.py --type text_prompt_qa --input {output_file}
```

#### Multimodal-prompt QA
Unlike text-prompt QA, multimodal-prompt QA requires an additional LLM-as-a-Judge process beforehand. If you use multiprocessing for this step, an aggregation process will likewise be necessary.

```shell
# LLM-as-a-Judge
sh scripts/LLM_as_a_Judge.sh

# Aggregate the results from each process
python src/output_merge.py --name {output_directory}

# Calculate the GPT-Score
python src/evaluation.py --type multimodal_prompt_qa --input {output_file}
```

---
## BibTex

---
## Acknowledgement
This dataset is constructed based on the [Yo’LLaVA](https://arxiv.org/abs/2406.09400) and [MC-LLaVA](https://arxiv.org/abs/2411.11706) datasets. Thank you for your outstanding work!

---
## License
This dataset is licensed under the [MIT License](./LICENSE). You are free to use, modify, and distribute this dataset under the terms of the MIT License.