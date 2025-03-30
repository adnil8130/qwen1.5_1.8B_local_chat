import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread

def test_qwen_infer():
    device = "cuda"  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-1.8B-Chat",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def test_qwen_stream():
    device = "cuda"  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-1.8B-Chat",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

    prompt = "你是谁？"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Use Thread to run generation in background
    # Otherwise, the process is blocked until generation is complete
    # and no streaming effect can be observed.
    from threading import Thread
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        print("new_text:", new_text)
        generated_text += new_text
    print("all_text:", generated_text)


def qwen_chat(model, tokenizer, query, history=[], device='cuda'):
    history += [{'role': 'user', 'content': query}]

    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    history += [{'role': 'assistant', 'content': response}]
    return response, history


def qwen_stream_chat(model, tokenizer, query, history=[], device='cuda'):
    history += [{'role': 'user', 'content': query}]
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Use Thread to run generation in background
    # Otherwise, the process is blocked until generation is complete
    # and no streaming effect can be observed.
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

if __name__ == '__main__':
    print(test_qwen_infer())