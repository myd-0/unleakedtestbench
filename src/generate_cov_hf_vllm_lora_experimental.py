import os
import re
import json
import inspect
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm
from huggingface_hub import snapshot_download

from data_utils import read_jsonl, write_jsonl

DEFAULT_MODEL = 'Qwen/Qwen2.5-3B-Instruct'
DEFAULT_ADAPTER = 'muratt0/qwen25-3b-unitsyn-testgen-lora'
EXPERIMENTAL_SUFFIX = 'vllmexp'
VLLM_ONLY_MESSAGE = 'This experimental script supports the vllm backend only.'


def extract_function_names_from_completion(completion: str) -> list:
    function_pattern = r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    return re.findall(function_pattern, completion, re.MULTILINE)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--adapter-path', type=str, default=DEFAULT_ADAPTER,
                        help='HF LoRA adapter repo id used for vLLM LoRA requests.')
    parser.add_argument('--adapter-name', type=str, default=None,
                        help='Optional path-safe adapter label used in output filenames.')
    parser.add_argument('--dataset', type=str, default='ULT',
                        help='dataset name or path (e.g. ULT, ULT_Lite, PLT, or a .jsonl path)')
    parser.add_argument('--num_tests', type=int, default=5, help='number of tests generated per program')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for vLLM inference')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='number of GPUs for tensor parallelism')
    parser.add_argument('--max_context_length', type=int, default=4096, help='maximum context length for truncation')
    parser.add_argument('--backend', type=str, default='vllm', choices=['vllm'],
                        help='experimental inference backend, fixed to vllm')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='limit number of dataset samples (useful for quick tests)')
    parser.add_argument('--generation-config', type=str, default='auto', choices=['auto', 'none'],
                        help='Sampling defaults source: auto uses the model generation_config, none uses neutral defaults.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed forwarded to vLLM when supported.')
    parser.add_argument('--batch-invariant', action='store_true',
                        help='Set VLLM_BATCH_INVARIANT=1 for more stable batched comparisons when supported.')
    parser.add_argument('--deterministic-scheduling', action='store_true',
                        help='Set VLLM_ENABLE_V1_MULTIPROCESSING=0 for more deterministic offline scheduling.')
    parser.add_argument('--fair-compare', action='store_true',
                        help='Favor transformers-like comparisons by forcing batch_size=1 and deterministic scheduling.')
    return parser.parse_args()


def resolve_backend(args):
    if args.backend != 'vllm':
        raise ValueError(VLLM_ONLY_MESSAGE)
    print('[+] Using experimental vLLM LoRA backend')
    return 'vllm'


def resolve_model_source(source: str | None) -> str | None:
    if source is None:
        return None
    candidate = Path(source).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return source


def resolve_adapter_local_path(adapter_source: str) -> str:
    candidate = Path(adapter_source).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return snapshot_download(repo_id=adapter_source, token=os.getenv('HUGGINGFACE_TOKEN'))


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
        return f'{model_abbrv}__{adapter_label}__{EXPERIMENTAL_SUFFIX}'
    return f'{model_abbrv}__{EXPERIMENTAL_SUFFIX}'


def load_adapter_rank(adapter_local_path: str) -> int:
    config_path = Path(adapter_local_path) / 'adapter_config.json'
    if not config_path.exists():
        return 64
    config = json.loads(config_path.read_text())
    return max(int(config.get('r', 16)), 16)


def resolve_dataset_path(dataset_arg: str) -> Path:
    repo_root = Path(__file__).parent.parent
    datasets_dir = repo_root / 'datasets'

    dataset_aliases = {
        'TestBench': 'ULT',
        'ULT': 'ULT',
        'ULT_Lite': 'ULT_Lite',
        'PLT': 'PLT',
    }

    dataset_name = dataset_aliases.get(dataset_arg, dataset_arg)
    candidate = Path(dataset_name)

    if candidate.is_absolute():
        dataset_path = candidate
    elif candidate.suffix == '.jsonl':
        dataset_path = repo_root / candidate
        if not dataset_path.exists():
            dataset_path = datasets_dir / candidate.name
    else:
        dataset_path = datasets_dir / f'{dataset_name}.jsonl'

    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')

    return dataset_path


def truncate_conversation(messages, tokenizer, max_length):
    system_message = next((m for m in messages if m['role'] == 'system'), None)
    last_user_message = next((m for m in reversed(messages) if m['role'] == 'user'), None)

    truncated_messages = []
    if system_message:
        truncated_messages.append(system_message)
    recent_assistant = [m for m in reversed(messages) if m['role'] == 'assistant']
    if recent_assistant:
        truncated_messages.append(recent_assistant[0])
    if last_user_message:
        truncated_messages.append(last_user_message)

    return truncated_messages


def format_chat_template(tokenizer, messages):
    try:
        apply_chat_template = tokenizer.apply_chat_template
        kwargs = {
            'tokenize': False,
            'add_generation_prompt': True,
        }
        if 'enable_thinking' in inspect.signature(apply_chat_template).parameters:
            kwargs['enable_thinking'] = False
        return apply_chat_template(messages, **kwargs)
    except Exception:
        formatted = ''
        for msg in messages:
            role, content = msg['role'], msg['content']
            if role == 'system':
                formatted += f'System: {content}\n\n'
            elif role == 'user':
                formatted += f'User: {content}\n\n'
            elif role == 'assistant':
                formatted += f'Assistant: {content}\n\n'
        return formatted + 'Assistant: '


def strip_think_tags(text):
    if '</think>' in text:
        return text.split('</think>')[1]
    return text


def build_sampling_params(args, llm):
    from vllm import SamplingParams

    sampling_params = None
    if args.generation_config == 'auto' and hasattr(llm, 'get_default_sampling_params'):
        try:
            sampling_params = llm.get_default_sampling_params()
        except Exception as e:
            print(f'Could not load model default sampling params: {e}')

    if sampling_params is None:
        sampling_params = SamplingParams()

    sampling_params.max_tokens = args.max_tokens
    sampling_params.temperature = args.temperature

    return sampling_params


def sampling_params_summary(sampling_params):
    fields = [
        'temperature',
        'top_p',
        'top_k',
        'min_p',
        'max_tokens',
        'repetition_penalty',
        'stop',
        'stop_token_ids',
    ]
    return {field: getattr(sampling_params, field, None) for field in fields}


def build_conversation_log_entry(messages, generated_test, template_append, round_number):
    messages_for_log = [m.copy() for m in messages]
    messages_for_log.append({'role': 'assistant', 'content': generated_test})
    messages_for_log.append({'role': 'user', 'content': template_append})
    return {'round': round_number, 'messages_sent': messages_for_log, 'response': generated_test}


def prepare_prompts_for_batch(data_batch, prompt_template, system_message, tokenizer):
    prompts = []
    for data in data_batch:
        try:
            func_names = extract_function_names_from_completion(data['code'])
            if not func_names:
                print('No function name found in the code')
                continue
            func_name = func_names[0]
            prompt = prompt_template.format(
                lang='python', program=data['code'],
                description=data['prompt'], func_name=func_name
            )
            messages = [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt},
            ]
            formatted = format_chat_template(tokenizer, messages)
            prompts.append((formatted, messages, func_name, data['code'], data['prompt'], data['task_id']))
        except Exception as e:
            print(f'Error preparing prompt: {e}')
    return prompts


def testgeneration_vllm_batch(prepared_prompts, llm, sampling_params, tokenizer, lora_request, max_tokens=4096):
    if not prepared_prompts:
        return []

    truncated_prompts = []
    for prompt_data in prepared_prompts:
        prompt_text, messages, func_name, code, desc, task_id = prompt_data
        tokens = tokenizer.encode(prompt_text)
        if len(tokens) > max_tokens:
            print(f'Truncating prompt from {len(tokens)} to {max_tokens} tokens')
            truncated_text = tokenizer.decode(tokens[-max_tokens:])
            truncated_prompts.append((truncated_text, messages, func_name, code, desc, task_id))
        else:
            truncated_prompts.append(prompt_data)

    prompt_texts = [p[0] for p in truncated_prompts]
    outputs = llm.generate(prompt_texts, sampling_params, lora_request=lora_request)

    results = []
    for i, output in enumerate(outputs):
        _, messages, func_name, code, desc, task_id = truncated_prompts[i]
        generated_test = strip_think_tags(output.outputs[0].text)
        results.append({'func_name': func_name, 'code': code, 'test': generated_test,
                        'prompt': desc, 'task_id': task_id, 'messages': [m.copy() for m in messages]})
    return results


def testgeneration_multiround_vllm(args, dataset, prompt_template, system_message, tokenizer, llm, lora_request, sampling_params, checkpoint_path=None):
    all_results = []
    template_append = ('Generate another test method for the function under test. '
                       'Your answer must be different from previously-generated test cases, '
                       'and should cover different statements and branches.')

    for batch_start in tqdm(range(0, len(dataset), args.batch_size), desc='Round 1'):
        batch = dataset[batch_start:batch_start + args.batch_size]
        prepared = prepare_prompts_for_batch(batch, prompt_template, system_message, tokenizer)
        results = testgeneration_vllm_batch(prepared, llm, sampling_params, tokenizer, lora_request)
        for r in results:
            all_results.append({'func_name': r['func_name'], 'code': r['code'],
                                'tests': [r['test']], 'prompt': r['prompt'], 'task_id': r['task_id'],
                                'conversation_log': [build_conversation_log_entry(r['messages'], r['test'], template_append, 1)]})
        if checkpoint_path is not None:
            write_jsonl(all_results, checkpoint_path)

    for test_round in range(1, args.num_tests):
        print(f'Round {test_round + 1}/{args.num_tests}')
        for batch_start in tqdm(range(0, len(all_results), args.batch_size), desc=f'Round {test_round + 1}'):
            batch = all_results[batch_start:batch_start + args.batch_size]
            prepared = []
            for result in batch:
                messages = [
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': prompt_template.format(
                        lang='python', program=result['code'],
                        description=result['prompt'], func_name=result['func_name']
                    )}
                ]
                for prev_test in result['tests']:
                    messages.append({'role': 'assistant', 'content': prev_test})
                    messages.append({'role': 'user', 'content': template_append})

                total_len = len(tokenizer.encode(' '.join(m['content'] for m in messages)))
                if total_len > args.max_context_length:
                    messages = truncate_conversation(messages, tokenizer, args.max_context_length)

                formatted = format_chat_template(tokenizer, messages)
                prepared.append((formatted, [m.copy() for m in messages], result['func_name'], result['code'],
                                 result['prompt'], result['task_id']))

            if prepared:
                round_results = testgeneration_vllm_batch(prepared, llm, sampling_params, tokenizer, lora_request)
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

    model_abbrv = make_model_basename(args.model, make_adapter_label(args.adapter_path, args.adapter_name))
    dataset_suffix = 'full' if dataset_path.stem == 'ULT' else dataset_path.stem.lower()

    print('=' * 50)
    print(f'Model: {model_abbrv}  |  Backend: {backend}')
    print(f'Dataset: {dataset_path.name}')
    print('=' * 50)

    output_file = output_dir / f'TestBench_{model_abbrv}_{args.num_tests}_{dataset_suffix}.jsonl'
    checkpoint_file = output_dir / f'TestBench_{model_abbrv}_{args.num_tests}_{dataset_suffix}_partial.jsonl'
    if output_file.exists():
        print(f'Results for {model_abbrv} already exist, skipping...')
        raise SystemExit(0)

    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=os.getenv('HUGGINGFACE_TOKEN'),
        trust_remote_code=True,
    )

    adapter_local_path = resolve_adapter_local_path(args.adapter_path)
    adapter_label = make_adapter_label(args.adapter_path, args.adapter_name) or 'adapter'
    adapter_rank = load_adapter_rank(adapter_local_path)
    base_model_abbrv = args.model.split('/')[-1]

    if args.fair_compare:
        args.batch_size = 1
        args.deterministic_scheduling = True

    if args.batch_invariant:
        os.environ['VLLM_BATCH_INVARIANT'] = '1'
    if args.deterministic_scheduling:
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

    print(f'Number of samples: {len(dataset)}')
    print(f'Loaded adapter repo/path: {args.adapter_path}')
    print(f'Resolved adapter local path: {adapter_local_path}')
    print('vLLM LoRA enabled: True')
    print(f'generation_config source: {args.generation_config}')
    print(f'seed: {args.seed}')
    if args.batch_invariant:
        print('VLLM_BATCH_INVARIANT=1')
    if args.deterministic_scheduling:
        print('VLLM_ENABLE_V1_MULTIPROCESSING=0')
    if args.fair_compare:
        print('fair_compare=True (batch_size forced to 1)')

    model_context_length = 16384
    if hasattr(tokenizer, 'model_max_length'):
        model_context_length = min(tokenizer.model_max_length, model_context_length)

    llm_kwargs = {
        'model': args.model,
        'tensor_parallel_size': args.tensor_parallel_size,
        'trust_remote_code': True,
        'dtype': 'bfloat16',
        'max_model_len': model_context_length,
        'enable_lora': True,
        'max_loras': 1,
        'max_lora_rank': adapter_rank,
        'quantization': 'awq' if Path(f'./quantized/{base_model_abbrv}_awq').exists() else None,
    }
    llm_signature = inspect.signature(LLM).parameters
    if args.generation_config == 'auto' and 'generation_config' in llm_signature:
        llm_kwargs['generation_config'] = 'auto'
    if 'seed' in llm_signature:
        llm_kwargs['seed'] = args.seed

    llm = LLM(**llm_kwargs)

    sampling_params = build_sampling_params(args, llm)
    print(f'Effective sampling params: {sampling_params_summary(sampling_params)}')

    lora_request = LoRARequest(adapter_label, 1, adapter_local_path)

    testing_results = testgeneration_multiround_vllm(
        args, dataset, prompt_template, system_message, tokenizer, llm, lora_request, sampling_params, checkpoint_path=checkpoint_file
    )

    write_jsonl(testing_results, output_file)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    print(f'Results saved to {output_file}')
