from openai import OpenAI, RateLimitError
import argparse
import json
import backoff
import concurrent
import tqdm
import time
import google.generativeai as genai


class TextHelper:
    def __init__(self, text):
        self.text = text

class OutputHelper:
    def __init__(self, output):
        self.outputs = [TextHelper(output)]

@backoff.on_exception(backoff.expo, RateLimitError)
def get_completion_from_gpt(prompt, model_name, max_tokens, api_key, temperature):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def get_completion_from_gemeni(prompt, model_name, max_tokens, api_key, temperature):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
    )
    return response.text

class OpenAILLM:
    def __init__(self, model_name, api_key) -> None:
        self.model_name = model_name
        self.api_key = api_key
    
    def generate(self, prompts, sampling_param):
        results = []
        for prompt in tqdm.tqdm(prompts):
            while True:
                try:
                    result = get_completion_from_gpt(prompt=prompt, model_name=self.model_name, api_key=self.api_key, temperature=sampling_param.temperature, max_tokens=sampling_param.max_tokens)
                except Exception as e:
                    continue
                results.append(OutputHelper(result))
                break
        return results

class GoogleLLM:
    def __init__(self, model_name, api_key) -> None:
        self.model_name = model_name
        self.api_key = api_key
    
    def generate(self, prompts, sampling_param):
        results = []
        for prompt in tqdm.tqdm(prompts):
            while True:
                try:
                    result = get_completion_from_gemeni(prompt=prompt, model_name=self.model_name, api_key=self.api_key, temperature=sampling_param.temperature, max_tokens=sampling_param.max_tokens)
                except Exception as e:
                    continue
                results.append(OutputHelper(result))
                break
        return results