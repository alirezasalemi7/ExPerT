from expert_score.utils.json_parsing import parse_json_aspects
from vllm import SamplingParams

def _create_aspect_generation_prompt(example, tokenizer, max_length_output=1024):
    if tokenizer is not None:
        tokenized_text = tokenizer(example["generated_output"], return_tensors="pt")
        input_ids = tokenized_text["input_ids"][0][:max_length_output]    
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        tokens = example["generated_output"].split()
        decoded_text = " ".join(tokens[:max_length_output])
    decoded_text = decoded_text.replace('"', " ")

    return f"""You are a helpful assistant capable of infering different atomic aspects from a generated response to a given prompt. Your task is to analyze the provided prompt and the generated output, generate a JSON list of atomic aspects that are presented in the generated output for that prompt and the corresponding evidence from the generated output. This evidence is a list of one or more sentences from the generated output that are related to this aspect. Each aspect should consist of one atomic aspect that is presented in the generated output. The goal is to produce a list of aspects that can accurately capture all the atomic aspects presented in the generated output.
    
    # your task:

    Given the test prompt and the generated output, generate a JSON list of atomic aspects that are presented in the generated output and the corresponding evidence from the generated output. This evidence is a list of one or more sentences from the generated output that are related to this aspect. Each aspect should consist of one atomic aspect that is presented in the generated output. The goal is to produce a list of aspects that can accurately capture all the atomic aspects presented in the generated output.

    ## your inputs:
        - test prompt: a prompt that shows a user's request for generating content
        - generated output: the generated output for the given prompt
    

    ## Your output should be a valid JSON list in ```json``` block that contains the following keys:
        - aspect: the aspect that is presented in the generated output.
        - description: the description of the aspect.
        - evidence: a list of one or more sentences from the expected output that are related to this aspect. This list should not be an empty list and should contain at least one sentence.
    
    Now, you are given a test prompt and the corresponding generated output. Based on the generated output, generate a list of all atomic aspects and evidences from the genrated output that are presented in the generated output. This list should capture all the atomic aspects presented in the generated output. 
    
    Test prompt: {example["input"]}
    generated output: {decoded_text}
    
    output: ```json
    """


def extract_aspects_from_text(inputs, llm, tokenizer, max_length_output=1024, max_length_generation=8096, max_retries=100):
    redo_examples = []
    for example in inputs:
        prompt = _create_aspect_generation_prompt(example, tokenizer, max_length_output)
        redo_examples.append((example, prompt))
    temperature = 0.0
    aspects = {}
    attempt = 0
    while len(redo_examples) > 0:
        prompts = [prompt for example, prompt in redo_examples]
        sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_length_generation, stop="```")
        outputs = llm.generate(prompts, sampling_params)
        new_redo_examples = []
        for (example, prompt), output in zip(redo_examples, outputs):
            try:
                aspects[example['id']] = parse_json_aspects(output.outputs[0].text)
            except Exception as e:
                print(f"Error: {e}")
                new_redo_examples.append((example, prompt))
        redo_examples = new_redo_examples
        if temperature < 1.0:
            temperature += 0.1
        attempt += 1
        if attempt > max_retries:
            not_possible_generate = [example['id'] for example, prompt in redo_examples]
            raise Exception("Could not generate aspects for ids: " + str(not_possible_generate))
    return aspects