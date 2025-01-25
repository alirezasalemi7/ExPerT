from vllm import LLM
from transformers import AutoTokenizer
from expert_score.aspects.extraction import extract_aspects_from_text
from expert_score.matching.match import get_match_table
from expert_score.matching.utils import calculate_scores_for_batch
from expert_score.utils.api_llm import OpenAILLM, GoogleLLM
from typing import List

def expert(inputs: List[str], outputs: List[str], references: List[str], model_name="google/gemma-2-27b-it", cache_dir=None, max_generated_output_length=512, max_evaluator_length=8096, max_retries=100, ignore_on_fail=True, google_llm=False, openai_llm=False, api_key=None):
    """
    Calculates the ExPerT score for each input based on a list of inputs, generated outputs, 
    and expected outputs.

    Args:
        inputs (List[str]): 
            A list of input texts for the personalized task provided by the user.
        outputs (List[str]): 
            A list of generated outputs from the model for the personalized task provided by the user.
        references (List[str]): 
            A list of reference outputs for the personalized task provided by the user.
        model_name (str, optional): 
            The name of the model used in ExPerT for evaluation. Defaults to "google/gemma-2-27b-it".
        cache_dir (str, optional): 
            The directory to cache the model. Defaults to the default location for the `vllm` library.
        max_generated_output_length (int, optional): 
            The maximum length of the generated output to be considered. Defaults to 512.
        max_evaluator_length (int, optional): 
            The maximum context length of the evaluator LLM. Defaults to 8096.
        max_retries (int, optional): 
            The maximum number of retries to obtain evaluator responses if any out-of-format 
            outputs are generated. Defaults to 100.
        ignore_on_fail (bool, optional): 
            Whether to ignore aspects that cannot be matched due to the LLM's inability to 
            generate outputs in the required format. Defaults to True.
        google_llm (bool, optional): 
            Whether to use the Google LLM API. Defaults to False.
        openai_llm (bool, optional): 
            Whether to use the OpenAI LLM API. Defaults to False.
        api_key (str, optional): 
            The API key for the Google LLM or OpenAI LLM API. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing average scores (average_scores), per-query scores (per_query_scores), and per-query explanations (per_query_explanations).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    except:
        tokenizer = None
    assert len(inputs) == len(outputs) == len(references), "The number of inputs, generated outputs and expected outputs should be the same"
    assert len(inputs) > 0, "The number of inputs should be greater than 0"
    assert len(outputs) > 0, "The number of generated_outputs should be greater than 0"
    assert len(references) > 0, "The number of expected_outputs should be greater than 0"
    assert model_name is not None, "The model_name should not be None"
    assert not google_llm or not openai_llm, "Only one of google_llm and openai_llm should be True"
    if google_llm or openai_llm:
        assert api_key is not None, "The api_key should not be None if either google_llm or openai_llm is True"
    
    if openai_llm:
        llm = OpenAILLM(model_name, api_key)
    elif google_llm:
        llm = GoogleLLM(model_name, api_key)
    else:
        llm = LLM(model_name, download_dir=cache_dir, dtype="float16")
    proccessed_inputs = [
        {
            "id": str(i),
            "input": inputs[i],
            "generated_output": references[i],
        }
        for i in range(len(inputs))
    ]
    expected_output_aspects = extract_aspects_from_text(proccessed_inputs, llm, tokenizer, max_generated_output_length, max_evaluator_length, max_retries)
    proccessed_inputs = [
        {
            "id": str(i),
            "input": inputs[i],
            "generated_output": outputs[i],
        }
        for i in range(len(inputs))
    ]
    generated_outputs_aspects = extract_aspects_from_text(proccessed_inputs, llm, tokenizer, max_generated_output_length, max_evaluator_length, max_retries)
    match_table = get_match_table(expected_output_aspects, generated_outputs_aspects, llm, max_evaluator_length, max_retries, ignore_on_fail)
    scores = calculate_scores_for_batch(match_table)
    per_query_scores = [scores["per_query"][str(i)] for i in range(len(inputs))]
    per_query_match_tables = [match_table[str(i)] for i in range(len(inputs))]
    average_scores = scores["average"]
    return {
        "average_scores": average_scores,
        "per_query_scores": per_query_scores,
        "per_query_explanations": per_query_match_tables,
    }