"""
Code Completion on HumanEval

Usage: python complete.py --base_model SLAM-group/NewHope --pretraining_tp 1 --output_dir output --n_gpu 8
"""

import os
import shutil
import argparse
import regex as re
from tqdm import trange
from multiprocessing import set_start_method, Pool

import torch
from transformers import GenerationConfig, set_seed
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig

from human_eval.data import read_problems, write_jsonl, stream_jsonl


model = None
tokenizer = None

example_input = 'def is_odd(number: int) -> bool:\n    """ Check whether the given number is odd\n    >>> is_odd(3)\n    True\n    >>> is_odd(6)\n    False\n    """\n'
example_output = 'def is_odd(number: int) -> bool:\n    """ Check whether the given number is odd\n    >>> is_odd(3)\n    True\n    >>> is_odd(6)\n    False\n    """\n    return number % 2 == 1'

generation_config = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    top_p=0.9,
    num_beams=5,
    num_return_sequences=1,
    early_stopping=True,
    max_new_tokens=1024
)


def load_model_tokenizer_to_device(args, i):
    global model, tokenizer
    
    try:
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
        
        # If you wish faster inference, set `config.pretraining_tp` to 1, but at the cost of higher GPU memory usage
        # Reference: https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/llama2
        config = LlamaConfig.from_pretrained(args.base_model)
        config.pretraining_tp = args.pretraining_tp

        model = LlamaForCausalLM.from_pretrained(args.base_model, config=config, torch_dtype=torch.float16, device_map={'': i})
        model.config.use_cache = True
        model.eval()
        
    except BaseException as e:
        print(e)
        raise


def generate_one_completion(prompt, proc_id):
    prompt = f"<s> ### Instruction:\nComplete the given function below:\n\n{example_input}\n\n### Response:\n{example_output}</s>\
<s> ### Instruction:\nComplete the given function below:\n\n{prompt}\n\n### Response:\n"
    
    try:
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(f"cuda:{proc_id}")
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
        
        completions = []
        for output in outputs:
            output = tokenizer.decode(output, skip_special_tokens=True)
            output = output.split('### Response:\n')[-1].strip()
            completions.append(output)
        return completions
    
    except BaseException as e:
        print(e)
        raise

def extract_code(completion: str):
    # If there is no "```", try matching by "import" statements or function definition
    if (start := completion.find('```')) == -1:
        # 1. First match "from ... import" or "import ..." where there is no function definition ahead
        if (match := re.search("(?<!def.+?)(?:from[^\n]+?import|import[^\n]+?\n)", completion, re.DOTALL)):
            start = match.span()[0]
        # 2. Then try matching by function definition
        elif match := re.search('def.+?:', completion):
            start = match.span()[0]
        else:
            return ''
        end = len(completion)
    # Otherwise try to find the coupled "```"
    else:
        start += 3
        if (end := completion.find('```', start + 1)) == -1:
            return ''

    code = completion[start: end].strip()
    # We have to consider the case "```Python\ndef..."
    if code.lower().startswith('python'):
        code = code[6:].strip()
    return code


def worker(args, proc_id, start, end, sub_problems):
    set_seed(args.generation_seed)
    
    load_model_tokenizer_to_device(args, proc_id)
    
    samples = []
    for i in trange(start, end, desc=f"#{proc_id}", position=proc_id):
        completions = generate_one_completion(sub_problems[i-start]['prompt'], proc_id=proc_id)
        codes = [extract_code(completion) for completion in completions]
        for code in codes:
            samples.append({
                'task_id': sub_problems[i-start]['task_id'],
                'completion': code
            })
    
    tmp_file = os.path.join(args.output_dir, "tmp", f"samples_{proc_id}.jsonl")
    write_jsonl(tmp_file, samples)


def main(args):
    if os.path.isdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.rmtree(os.path.join(args.output_dir, 'tmp'), ignore_errors=True)
    os.makedirs(os.path.join(args.output_dir, "tmp"), exist_ok=True)
    
    num_worker = args.n_gpu
    print(f"Starting {num_worker} workers for generation ... ")
    
    problems = read_problems()
    problem_list = []
    for k, v in problems.items():
        v['task_id'] = k
        problem_list.append(v)
    
    if num_worker > 1:
        pool = Pool(num_worker)
        for i in range(num_worker):
            start = i * len(problem_list) // num_worker
            end = (i + 1) * len(problem_list) // num_worker
            pool.apply_async(func=worker, args=[args, i, start, end, problem_list[start: end]])
        pool.close()
        pool.join()
    else:
        worker(args, 0, 0, len(problem_list), problem_list)
    
    samples = []
    for i in range(num_worker):
        tmp_file = os.path.join(args.output_dir, "tmp", f"samples_{i}.jsonl")
        for json_line in stream_jsonl(tmp_file):
            samples.append(json_line)
    
    shutil.rmtree(os.path.join(args.output_dir, "tmp"))
    final_file = os.path.join(args.output_dir, "samples.jsonl")
    write_jsonl(final_file, samples)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_model', type=str, default='SLAM-group/NewHope',
        help="model name or path"
    )
    parser.add_argument(
        '--pretraining_tp', type=int, default=1,
        help='pretraining_tp used in llama2, see https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/llama2'
    )
    parser.add_argument(
        '--output_dir', type=str, default='output',
        help='output directory of "samples.jsonl" used for HumanEval'
    )
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument(
        '--n_gpu', type=int, default=8,
        help="how many gpus will be used for concurrent code completion"
    )
    parser.add_argument('--generation_seed', type=int, default=42)
    args = parser.parse_args()
    
    set_start_method('spawn', force=True)
    
    main(args)
    