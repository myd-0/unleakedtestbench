import os
import re
import json
import inspect
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import torch

from data_utils import read_jsonl, write_jsonl, add_lineno

ADAPTER_TRANSFORMERS_ONLY_MESSAGE = "Adapter runs currently support the transformers backend only."

DEFAULT_MODEL = 'codellama/CodeLlama-7b-Instruct-hf'

model_list = [
    # "codellama/CodeLlama-7b-Instruct-hf",
    # "ByteDance-Seed/Seed-Coder-8B-Instruct",
    "google/gemma-3-4b-it",
    # "google/gemma-3-12b-it",
    # "google/gemma-3-27b-it",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    # "Qwen/Qwen2.5-Coder-14B-Instruct",
    # "Qwen/Qwen2.5-Coder-32B-Instruct",
    # 'deepseek-ai/deepseek-coder-1.3b-instruct',
    'deepseek-ai/deepseek-coder-6.7b-instruct',
    'deepseek-ai/deepseek-coder-33b-instruct',
    # "microsoft/Phi-4-mini-instruct",
]


def extract_function_names_from_completion(completion: str) -> list:
    function_pattern = r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    return re.findall(function_pattern, completion, re.MULTILINE)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default="ULT",
                        help="dataset name or path (e.g. ULT, ULT_Lite, PLT, or a .jsonl path)")
    parser.add_argument("--num_tests", type=int, default=5, help='number of tests generated per program')
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=128, help='batch size for vLLM inference')
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help='number of GPUs for tensor parallelism')
    parser.add_argument("--max_context_length", type=int, default=4096, help='maximum context length for truncation')
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "vllm", "transformers"],
                        help='inference backend: auto (detect), vllm (CUDA only), transformers (MPS/CPU)')
    parser.add_argument("--max_samples", type=int, default=None,
                        help='limit number of dataset samples (useful for quick tests)')
    parser.add_argument("--adapter-path", type=str, default=None,
                        help='Optional PEFT adapter path or Hugging Face repo id.')
    parser.add_argument("--adapter-name", type=str, default=None,
                        help='Optional path-safe adapter label used in output filenames.')
    return parser.parse_args()


def resolve_backend(args):
    if args.adapter_path:
        if args.backend in {"auto", "transformers"}:
            if args.backend == "auto":
                print("[+] Adapter detected — forcing transformers backend")
            return "transformers"
        raise ValueError(ADAPTER_TRANSFORMERS_ONLY_MESSAGE)

    if args.backend == "auto":
        if torch.cuda.is_available():
            print("[+] CUDA detected — using vLLM backend")
            return "vllm"
        elif torch.backends.mps.is_available():
            print("[+] Apple MPS detected — using transformers backend")
            return "transformers"
        else:
            print("[+] No GPU detected — using transformers backend on CPU")
            return "transformers"
    return args.backend


def resolve_model_source(source: str | None) -> str | None:
    if source is None:
        return None
    candidate = Path(source).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return source


def make_adapter_label(adapter_path: str | None, adapter_name: str | None) -> str | None:
    if adapter_path is None:
        return None
    if adapter_name:
        raw = adapter_name
    else:
        raw = adapter_path.rstrip('/').split('/')[-1]
    safe = re.sub(r'[^A-Za-z0-9._-]+', '-', raw).strip('-')
    return safe or 'adapter'


def make_model_basename(model_name: str, adapter_label: str | None) -> str:
    model_abbrv = model_name.split('/')[-1]
    if adapter_label:
        return f"{model_abbrv}__{adapter_label}"
    return model_abbrv


def has_lora_modules(model) -> bool:
    return any("lora_A" in name or "lora_B" in name for name, _ in model.named_modules())


def resolve_dataset_path(dataset_arg: str) -> Path:
    repo_root = Path(__file__).parent.parent
    datasets_dir = repo_root / "datasets"

    dataset_aliases = {
        "TestBench": "ULT",
        "ULT": "ULT",
        "ULT_Lite": "ULT_Lite",
        "PLT": "PLT",
    }

    dataset_name = dataset_aliases.get(dataset_arg, dataset_arg)
    candidate = Path(dataset_name)

    if candidate.is_absolute():
        dataset_path = candidate
    elif candidate.suffix == ".jsonl":
        dataset_path = repo_root / candidate
        if not dataset_path.exists():
            dataset_path = datasets_dir / candidate.name
    else:
        dataset_path = datasets_dir / f"{dataset_name}.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return dataset_path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def truncate_conversation(messages, tokenizer, max_length):
    system_message = next((m for m in messages if m["role"] == "system"), None)
    last_user_message = next((m for m in reversed(messages) if m["role"] == "user"), None)

    truncated_messages = []
    if system_message:
        truncated_messages.append(system_message)
    recent_assistant = [m for m in reversed(messages) if m["role"] == "assistant"]
    if recent_assistant:
        truncated_messages.append(recent_assistant[0])
    if last_user_message:
        truncated_messages.append(last_user_message)

    return truncated_messages


def format_chat_template(tokenizer, messages):
    try:
        apply_chat_template = tokenizer.apply_chat_template
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "enable_thinking" in inspect.signature(apply_chat_template).parameters:
            kwargs["enable_thinking"] = False
        return apply_chat_template(messages, **kwargs)
    except Exception:
        formatted = ""
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        return formatted + "Assistant: "


def strip_think_tags(text):
    if "</think>" in text:
        return text.split("</think>")[1]
    return text


def build_conversation_log_entry(messages, generated_test, template_append, round_number):
    messages_for_log = [m.copy() for m in messages]
    messages_for_log.append({"role": "assistant", "content": generated_test})
    messages_for_log.append({"role": "user", "content": template_append})
    return {"round": round_number, "messages_sent": messages_for_log, "response": generated_test}


def prepare_prompts_for_batch(data_batch, prompt_template, system_message, tokenizer):
    prompts = []
    for data in data_batch:
        try:
            func_names = extract_function_names_from_completion(data["code"])
            if not func_names:
                print("No function name found in the code")
                continue
            func_name = func_names[0]
            prompt = prompt_template.format(
                lang='python', program=data['code'],
                description=data['prompt'], func_name=func_name
            )
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            formatted = format_chat_template(tokenizer, messages)
            prompts.append((formatted, messages, func_name, data['code'], data['prompt'], data['task_id']))
        except Exception as e:
            print(f"Error preparing prompt: {e}")
    return prompts


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

def testgeneration_vllm_batch(prepared_prompts, llm, sampling_params, tokenizer, max_tokens=4096):
    if not prepared_prompts:
        return []

    truncated_prompts = []
    for prompt_data in prepared_prompts:
        prompt_text, messages, func_name, code, desc, task_id = prompt_data
        tokens = tokenizer.encode(prompt_text)
        if len(tokens) > max_tokens:
            print(f"Truncating prompt from {len(tokens)} to {max_tokens} tokens")
            truncated_text = tokenizer.decode(tokens[-max_tokens:])
            truncated_prompts.append((truncated_text, messages, func_name, code, desc, task_id))
        else:
            truncated_prompts.append(prompt_data)

    prompt_texts = [p[0] for p in truncated_prompts]
    outputs = llm.generate(prompt_texts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        _, messages, func_name, code, desc, task_id = truncated_prompts[i]
        generated_test = strip_think_tags(output.outputs[0].text)
        results.append({'func_name': func_name, 'code': code, 'test': generated_test,
                        'prompt': desc, 'task_id': task_id, 'messages': [m.copy() for m in messages]})
    return results


def testgeneration_multiround_vllm(args, dataset, prompt_template, system_message, tokenizer, llm, checkpoint_path=None):
    from vllm import SamplingParams
    all_results = []
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, top_p=1.0)
    template_append = ("Generate another test method for the function under test. "
                       "Your answer must be different from previously-generated test cases, "
                       "and should cover different statements and branches.")

    for batch_start in tqdm(range(0, len(dataset), args.batch_size), desc="Round 1"):
        batch = dataset[batch_start:batch_start + args.batch_size]
        prepared = prepare_prompts_for_batch(batch, prompt_template, system_message, tokenizer)
        results = testgeneration_vllm_batch(prepared, llm, sampling_params, tokenizer)
        for r in results:
            all_results.append({'func_name': r['func_name'], 'code': r['code'],
                                'tests': [r['test']], 'prompt': r['prompt'], 'task_id': r['task_id'],
                                'conversation_log': [build_conversation_log_entry(r['messages'], r['test'], template_append, 1)]})
        if checkpoint_path is not None:
            write_jsonl(all_results, checkpoint_path)

    for test_round in range(1, args.num_tests):
        print(f"Round {test_round + 1}/{args.num_tests}")
        for batch_start in tqdm(range(0, len(all_results), args.batch_size), desc=f"Round {test_round + 1}"):
            batch = all_results[batch_start:batch_start + args.batch_size]
            prepared = []
            for result in batch:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_template.format(
                        lang='python', program=result['code'],
                        description=result['prompt'], func_name=result['func_name']
                    )}
                ]
                for prev_test in result['tests']:
                    messages.append({"role": "assistant", "content": prev_test})
                    messages.append({"role": "user", "content": template_append})

                total_len = len(tokenizer.encode(" ".join(m["content"] for m in messages)))
                if total_len > args.max_context_length:
                    messages = truncate_conversation(messages, tokenizer, args.max_context_length)

                formatted = format_chat_template(tokenizer, messages)
                prepared.append((formatted, [m.copy() for m in messages], result['func_name'], result['code'],
                                 result['prompt'], result['task_id']))

            if prepared:
                round_results = testgeneration_vllm_batch(prepared, llm, sampling_params, tokenizer)
                for i, new_result in enumerate(round_results):
                    idx = batch_start + i
                    if idx < len(all_results):
                        all_results[idx]['tests'].append(new_result['test'])
                        all_results[idx].setdefault('conversation_log', []).append(
                            build_conversation_log_entry(new_result['messages'], new_result['test'], template_append, test_round + 1)
                        )
                if checkpoint_path is not None:
                    write_jsonl(all_results, checkpoint_path)

    return all_results


# ---------------------------------------------------------------------------
# Transformers backend (M2 / CPU)
# ---------------------------------------------------------------------------

def generate_one_transformers(model, tokenizer, messages, args, device):
    prompt_text = format_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                       max_length=args.max_context_length).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=max(args.temperature, 0.01) if args.temperature > 0 else None,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return strip_think_tags(tokenizer.decode(new_tokens, skip_special_tokens=True))


def testgeneration_multiround_transformers(args, dataset, prompt_template, system_message, tokenizer, model, checkpoint_path=None):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    template_append = ("Generate another test method for the function under test. "
                       "Your answer must be different from previously-generated test cases, "
                       "and should cover different statements and branches.")

    all_results = []

    for data in tqdm(dataset, desc="Round 1"):
        func_names = extract_function_names_from_completion(data["code"])
        if not func_names:
            continue
        func_name = func_names[0]
        prompt = prompt_template.format(
            lang='python', program=data['code'],
            description=data['prompt'], func_name=func_name
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        try:
            test = generate_one_transformers(model, tokenizer, messages, args, device)
        except Exception as e:
            print(f"Error generating test for {func_name}: {e}")
            test = f"# Error: {e}"
        all_results.append({'func_name': func_name, 'code': data['code'],
                            'tests': [test], 'prompt': data['prompt'], 'task_id': data['task_id'],
                            'conversation_log': [build_conversation_log_entry(messages, test, template_append, 1)]})
        if checkpoint_path is not None:
            write_jsonl(all_results, checkpoint_path)

    for test_round in range(1, args.num_tests):
        print(f"Round {test_round + 1}/{args.num_tests}")
        for result in tqdm(all_results, desc=f"Round {test_round + 1}"):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_template.format(
                    lang='python', program=result['code'],
                    description=result['prompt'], func_name=result['func_name']
                )}
            ]
            for prev_test in result['tests']:
                messages.append({"role": "assistant", "content": prev_test})
                messages.append({"role": "user", "content": template_append})

            try:
                test = generate_one_transformers(model, tokenizer, messages, args, device)
            except Exception as e:
                print(f"Error in round {test_round + 1} for {result['func_name']}: {e}")
                test = f"# Error: {e}"
            result['tests'].append(test)
            result.setdefault('conversation_log', []).append(
                build_conversation_log_entry(messages, test, template_append, test_round + 1)
            )
            if checkpoint_path is not None:
                write_jsonl(all_results, checkpoint_path)

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    backend = resolve_backend(args)

    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    dataset_path = resolve_dataset_path(args.dataset)
    dataset = read_jsonl(dataset_path)
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]

    prompt_template = open('prompt/template_base.txt').read()
    system_template = open('prompt/system.txt').read()
    system_message = system_template.format(lang='python')

    # If user passed a specific model via CLI, run only that model.
    # Otherwise run the full model_list.
    if args.model != DEFAULT_MODEL:
        models_to_run = [args.model]
    else:
        models_to_run = model_list

    for model_name in models_to_run:
        args.model = model_name
        adapter_label = make_adapter_label(args.adapter_path, args.adapter_name)
        model_abbrv = make_model_basename(args.model, adapter_label)
        dataset_suffix = 'full' if dataset_path.stem == 'ULT' else dataset_path.stem.lower()
        print('=' * 50)
        print(f'Model: {model_abbrv}  |  Backend: {backend}')
        print(f'Dataset: {dataset_path.name}')
        print('=' * 50)

        output_file = output_dir / f'TestBench_{model_abbrv}_{args.num_tests}_{dataset_suffix}.jsonl'
        checkpoint_file = output_dir / f'TestBench_{model_abbrv}_{args.num_tests}_{dataset_suffix}_partial.jsonl'
        if output_file.exists():
            print(f"Results for {model_abbrv} already exist, skipping...")
            continue

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                token=os.getenv("HUGGINGFACE_TOKEN"),
                trust_remote_code=True,
            )

            print(f"Number of samples: {len(dataset)}")

            if backend == "vllm":
                from vllm import LLM

                model_context_length = 16384
                if hasattr(tokenizer, 'model_max_length'):
                    model_context_length = min(tokenizer.model_max_length, model_context_length)

                llm = LLM(
                    model=args.model,
                    tensor_parallel_size=args.tensor_parallel_size,
                    trust_remote_code=True,
                    dtype="float16",
                    max_model_len=model_context_length,
                    quantization="awq" if Path(f"./quantized/{model_abbrv}_awq").exists() else None,
                )

                testing_results = testgeneration_multiround_vllm(
                    args, dataset, prompt_template, system_message, tokenizer, llm, checkpoint_path=checkpoint_file
                )

                del llm
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            else:  # transformers
                if torch.backends.mps.is_available():
                    dtype = torch.float16
                    device_map = None  # will move manually in the function
                elif torch.cuda.is_available():
                    dtype = torch.float16
                    device_map = "auto"
                else:
                    dtype = torch.float32
                    device_map = None

                hf_model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    token=os.getenv("HUGGINGFACE_TOKEN"),
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map=device_map,
                )

                if args.adapter_path:
                    from peft import PeftModel
                    hf_model = PeftModel.from_pretrained(
                        hf_model,
                        resolve_model_source(args.adapter_path),
                        token=os.getenv("HUGGINGFACE_TOKEN"),
                    )
                    print(f"Loaded adapter: {resolve_model_source(args.adapter_path)}")
                    print(f"LoRA modules present: {has_lora_modules(hf_model)}")
                else:
                    print("Loaded base model only.")

                testing_results = testgeneration_multiround_transformers(
                    args, dataset, prompt_template, system_message, tokenizer, hf_model, checkpoint_path=checkpoint_file
                )

                del hf_model

            write_jsonl(testing_results, output_file)
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            print(f"Results saved to {output_file}")

        except Exception as e:
            print(f"Error with model {model_abbrv}: {e}")
            raise

        del tokenizer
        print(f"Completed: {model_abbrv}")
