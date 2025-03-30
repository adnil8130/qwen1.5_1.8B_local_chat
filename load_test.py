import aiohttp
import asyncio
import json
import random
import logging
import time
from typing import List, Tuple
from modelscope import AutoModelForCausalLM, AutoTokenizer

import numpy as np

logger = logging.getLogger(__name__)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
API_URL = 'http://127.0.0.1:8000/chat'

HEADERS = {
    'Content-Type': 'application/json'
}


async def send_request(session, payload, prompt_len, tokenizer):
    request_start_time = time.time()
    async with session.post(API_URL, json=payload, headers=HEADERS) as response:
        if response.status == 200:
            result = await response.json()
            content = result['choices'][-1]['message']['content']
            encoded_input = tokenizer.encode(content)
            completion_tokens = len(encoded_input)
            request_end_time = time.time()
            request_latency = request_end_time - request_start_time
            REQUEST_LATENCY.append((prompt_len, completion_tokens, request_latency))
            return result
        else:
            return {'error': response.status, 'message': await response.text()}


async def benchmark(
    input_requests: List[Tuple[str, int, int]],
    tokenizer
) -> None:
    async with aiohttp.ClientSession() as session:
        for idx, request in enumerate(input_requests):
            prompt, prompt_len, output_len = request
            payload = {
                "model": "string",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0,
                "top_p": 0,
                "max_length": 0,
                "stream": False
            }
            response = await send_request(session, payload, prompt_len, tokenizer)
            print(f"Response {idx + 1}: {json.dumps(response, ensure_ascii=False, indent=2)}")


def sample_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: "PreTrainedTokenizerBase",
) -> List[Tuple[str, int, int]]:
    # Load the dataset
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]
    # Tokenize the prompts and completions(input_msg, input_token_len, output_token_len).
    tokenized_dataset = []
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))
    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))
    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def main():
    logger.info("Preparing for benchmark.")
    dataset_path = r'ShareGPT_V3_unfiltered_cleaned_split.json'

    num_request = 25
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
    input_requests = sample_requests(dataset_path, num_request, tokenizer)

    logger.info("Benchmark starts.")
    benchmark_start_time = time.time()
    asyncio.run(benchmark(input_requests, tokenizer))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency in REQUEST_LATENCY
        ]
    )
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    )
    print("Average latency per output token: " f"{avg_per_output_token_latency:.2f} s")

if __name__ == '__main__':
    # prompt = """
    #         你是谁？
    #          """
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
    # asyncio.run(benchmark([(prompt, 3, 3)], tokenizer))
    main()
