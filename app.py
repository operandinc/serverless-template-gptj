from transformers import GPTJForCausalLM, GPT2Tokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    print("loading to CPU...")
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Prompt is a required argument.
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Tokenize inputs
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    token_count = input_tokens.size(dim=1)

    # (Optional) Max Token Length
    max_tokens = model_inputs.get('max_tokens', 256)
    if token_count + max_tokens  > 2048:
        return {'message': "Max token length exceeded"}

    # Optional: Temperature
    temperature = model_inputs.get('temperature', 0.8)

    # Optional: Top P
    top_p = model_inputs.get('top_p', 0.7)

    # Optional: Top K
    top_k = model_inputs.get('top_k', 0)

    # Optional: Number of Generations
    n = model_inputs.get('n', 1)

    # Optional: Return the prompt in the response
    include_prompt = model_inputs.get('include_prompt', False)

    # Optional: Stop sequences
    stop_sequences = model_inputs.get('stop_sequences', [])

    # Perform the inference.
    outputs = []
    for _ in range(n):
        output = model.generate(
            input_tokens,
            max_length=token_count + max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )
        text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

        # If we don't want to include the prompt in the output, remove it.
        if not include_prompt:
            text = text[len(prompt):]

        # If we have any stop sequences, truncate the output.
        for stop in stop_sequences:
            if stop in text:
                text = text[:text.index(stop)]

        # Finally, we can append this as a valid output.
        outputs.append(text)

    # Return the outputs as a JSON object.
    return {'outputs': outputs}
