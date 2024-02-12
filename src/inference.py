import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_inference(model_path):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token="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",
    )
    # to be able to tokenize text in batches
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", token="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"
    )

    model.eval()
    return model, tokenizer


def inference(
    prompt,
    model,
    tokenizer,
    n_samples,
    num_beams,
    temperature,
    max_length,
    top_p,
    max_time,
):

    if isinstance(prompt, str):
        prompt = [prompt]
    do_sample = True if top_p else False

    # Make n_samples of the prompt
    prompt *= n_samples

    # Tokenize
    encoded_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(
        model.device
    )

    with torch.no_grad():
        encoded_outputs = model.generate(
            **encoded_inputs,
            max_length=max_length,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=num_beams,
            top_p=top_p,
            max_time=max_time,
        )
    texts = tokenizer.batch_decode(encoded_outputs)
    return texts
