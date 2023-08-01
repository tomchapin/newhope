# NewHope: Harnessing 99% of GPT-4's Programming Capabilities

We introduce NewHope, a fine-tuned chat model based on llama-2-13b, aiming to provide a strong coding capability. NewHope handle different languages including Python, C++, Java, JavaScript, Go, and more. Preliminary evaluation on HumanEval shows that **NewHope possesses 99% of GPT-4's programming capabilities**.

**Contact**: SLAM (<ins>S</ins>UFE <ins>L</ins>arge <ins>A</ins>I <ins>M</ins>odel) is a research group at Shanghai University of Finance and Economics. 
cui.wanyun@sufe.edu.cn 

**TODO**: We will release more evaluatation results and training details later.

# Evaluation Results

We evaluated NewHope on [HumanEval](https://github.com/openai/human-eval) using the official evaluation script by OpenAI. We compared the Pass@1 metric of NewHope with other models. The results of other models are from PapersWithCode.

| Model | Pass@1 |
| ----- | ------ |
| **GPT-4** | **67.0**   |
| **NewHope** | **66.5**  | 
| PanGu-Coder2 15B | 61.6   |
| WizardCoder 15B | 57.3  |
| phi-1 1.3B | 50.6 |
| GPT-3.5 | 48.1 |
| phi-1-small | 45.0 |
| PaLM-Coder | 36.0 |
| CodeGeeX2-6B | 35.9 |

# Model Weights

We have open-sourced the model weights [NewHope](https://huggingface.co/SLAM-group).

We are uploading the model weights. The weights will be available in a few hours.


# Usage

To load the NewHope model using Transformers, use the following code:
```
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

base_model = ""
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="auto")
# model.config.use_cache is default to `False`. For inference: `model.config.use_cache = True`
```
**Note:** At least Huggingface Transformers **4.31.0** is required to load this model!

You can ask NewHope to generate code with instructions. We provide a simple example of how NewHope model generates code with the specific prompt:
```
# Suppose required tokenizer and model have already been loaded

instruction = "Write a Python function to tell me what the date is today."
prompt = f"<s> ### Instruction:\n{instruction}\n\n### Response:\n"
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
output = model.generate(**inputs, do_sample=True, top_p=0.9, max_new_tokens=2048)[0]
decoded_output = tokenizer.decode(output, skip_special_tokens=True).split("### Response:\n")[-1].strip()
print(decoded_output)
```

You can also interact with NewHope in a dialog manner with the following prompt:
```
<s> ### Instruction:\nQ1\n\n### Response:\nA1</s><s> ### Instruction:\nQ2\n\n### Response:\nA2</s>
```


# Evaluation

### Local setup
1. Install HumanEval for evaluation. [Details](https://github.com/openai/human-eval)
2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

---
For HumanEval, we use the following prompt:
```
example_input = 'def is_odd(number: int) -> bool:\n    """ Check whether the given number is odd\n    >>> is_odd(3)\n    True\n    >>> is_odd(6)\n    False\n    """\n'
example_output = 'def is_odd(number: int) -> bool:\n    """ Check whether the given number is odd\n    >>> is_odd(3)\n    True\n    >>> is_odd(6)\n    False\n    """\n    return number % 2 == 1'

task_in_humaneval = "REPLACE `task_in_humaneval` WITH THE SPECIFIC TASK IN HUMANEVAL DATA"

prompt = f"<s> ### Instruction:\nComplete the given function below:\n\n{example_input}\n\n### Response:\n{example_output}</s><s> ### Instruction:\nComplete the given function below:\n\n{task_in_human_eval}\n\n### Response:\n"
```

To reproduce the results on HumanEval, use the following script:
```
python complete_newhope.py --base_model llama2 --output_dir output --n_gpu 8
```
The above script will generate `samples.jsonl` in `output_dir`, which can be directly evaluated by HumanEval. [Evaluation procedure](https://github.com/openai/human-eval). We conducted the experiment with `fp16` on 8xA800, 80GB GPUs, reaching `66.5%` on Pass@1 (v.s. GPT4 `67.0%`).

# Citation

```
@misc{2023newhope,
    title={NewHope: Harnessing 99% of GPT-4's Programming Capabilities},
    author={NewHope team},
    howpublished = https://github.com/SLAM-group/newhope,
    year={2023}
}
```

