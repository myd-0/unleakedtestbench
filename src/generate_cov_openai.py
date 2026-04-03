#baseline for targeted line coverage: not providing the target line number
import os
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import openai
import json
from openai import OpenAI
openai.api_key=os.getenv("OPENAI_API_KEY") #personal key

client=OpenAI(api_key=openai.api_key)

from data_utils import read_jsonl, write_jsonl, add_lineno


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='TestBench')
    parser.add_argument("--lang", type=str, default='python')
    parser.add_argument("--model", type=str, default='gpt-4o', choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o',"gpt-4o-mini","gpt-4.1-mini","claude-3-5-haiku-latest"])
    parser.add_argument("--num_tests", type=int, default=5, help='number of tests generated per program')
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=None,
                        help='limit number of dataset samples (useful for quick tests)')
    return parser.parse_args()


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
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return dataset_path

def extract_function_names_from_completion(completion: str) -> list:
    """Extract function names from the completion code."""
    import re
    # Regular expression to match function definitions (ignoring indented functions)
    function_pattern = r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    
    # Find all matches
    function_names = re.findall(function_pattern, completion, re.MULTILINE)
    return function_names

def generate_completion(args,prompt,system_message=''):
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    code_output=response.choices[0].message.content
    return code_output


def testgeneration_multiround(args,prompt,system_message=''):
    """generate test cases with multi-round conversation, each time generate one test case"""
    template_append="Generate another test method for the function under test. Your answer must be different from previously-generated test cases, and should cover different statements and branches."
    generated_tests=[]
    conversation_log=[]
    messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    try:
        for i in range(args.num_tests):
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            generated_test=response.choices[0].message.content
            messages.append({"role": "assistant", "content": generated_test})
            messages.append({"role": "user", "content": template_append})

            conversation_log.append({"round": i+1, "messages_sent": [m.copy() for m in messages], "response": generated_test})
            generated_tests.append(generated_test)
            print(generated_test)
    except Exception as e:
        print("Error in generating test cases:", e)
        generated_tests.append(f"Error in generating test cases: {e}")
    return generated_tests, conversation_log


lang_exts={'python':'py', 'java':'java', 'c++':'cpp'}


if __name__=='__main__':
    args=parse_args()
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    dataset_path = resolve_dataset_path(args.dataset)
    dataset = read_jsonl(dataset_path)
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]

    prompt_template=open('prompt/template_base.txt').read()
    system_template=open('prompt/system.txt').read()
    system_message=system_template.format(lang='python')

    model_abbrv = args.model
    dataset_suffix = 'full' if dataset_path.stem == 'ULT' else dataset_path.stem.lower()
    output_file = output_dir / f'TestBench_{model_abbrv}_{args.num_tests}_{dataset_suffix}.jsonl'

    print(f'Model: {args.model}')
    print(f'Dataset: {dataset_path.name}')
    print(f'Number of samples: {len(dataset)}')

    testing_results=[]

    for i in tqdm(range(len(dataset))):
        data=dataset[i]
        try:
            func_name = data['func_name']
            desc=data['prompt']
            code=data['code']

            prompt=prompt_template.format(lang='python', program=code, description=desc, func_name=func_name)
            generated_tests, conversation_log=testgeneration_multiround(args,prompt,system_message)

            testing_data={'func_name':func_name,'code':code,'tests':generated_tests,'prompt':desc,'task_id':data['task_id'],'conversation_log':conversation_log}
            testing_results.append(testing_data)
        except Exception as e:
            print(f"Error processing task {i}: {e}")

        write_jsonl(testing_results, output_file)

    write_jsonl(testing_results, output_file)
    print(f"Results saved to {output_file}")
