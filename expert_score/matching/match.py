from vllm import SamplingParams
from expert_score.utils.json_parsing import parse_json_aspect_matching, parse_json_content_style_matching

def create_aspect_matching_prompt(aspect, list_of_aspects):
    list_of_aspects = "\n\n".join([
        f"aspect title: {x['aspect']}\naspect detail: {x['description']}" for x in list_of_aspects
    ])
    return f"""You are a helpful assistant with expertise in judging if two concepts match each other based on their title and description. Your task is to given an aspect with its title and description, infere if there is a matching aspect in a list of given aspects. If you could find a match, you should provide the aspect that matches the most with the given aspect. Otherwise, you should not match. You can use the title and description of the aspects to make the judgement. If the title and description of the two aspects are similar and about the same concept, then they are matched. Otherwise, they are not matched.

    # Your task:

    Given an aspect with its title and description and a list of other aspects, you should find the aspect that matches the most with the given aspect. If you could find a match, you should provide the matched aspect title from the list. Otherwise, you should not match.

    ## Your inputs:
        - test aspect: the aspect that you should find a match for it
            - aspect title: the aspect that you should find a match for it
            - aspect detail: the description of the aspect that you should find a match for it
        - list of aspects: a list of aspects that you should find a match for the test aspect from them
            - aspect title: the title of the aspect
            - aspect detail: the description of the aspect
        
    ## Your output should be a valid JSON list in ```json``` block that contains the following keys:
        - reason: a string that shows the reason of the match or not match
        - is_matched: a boolean value that shows if the test aspect is matched with any of the aspects in the list. If it is matched, the value should be True, otherwise, it should be false.
        - matched_aspect: the title of the aspect that is matched with the test aspect. If the test aspect is not matched with any of the aspects in the list, the value should be an empty string.
        
    # test aspect

    aspect title: {aspect['aspect']}
    aspect detail: {aspect['description']}

    # list of aspects
    {list_of_aspects}

    # output: ```json
    """

def create_content_matching_prompt(aspect_1, aspect_2,):
    text_snippets_1 = "\n".join(aspect_1['evidence'])
    text_snippets_2 = "\n".join(aspect_2['evidence'])
    return f"""You are a helpful assistant and an impartial judge, tasked with assessing if two lists of text snippets match in content and intent around a given aspect. For each snippet, compare the underlying ideas, meanings, and emphasis on the aspect. Ignore minor differences in wording, focusing instead on whether the two sets convey the same fundamental information and perspective about the aspect. 

    # Your task:
    Given an aspect and two lists of text snippets, you should determine if the two lists match in content and intent around the aspect. If the two lists are similar in terms of the underlying ideas, meanings, and emphasis on the aspect, then they are matched. Otherwise, they are not matched.

    ## Your inputs:
        - aspect: the aspect that the two lists of text snippets are about
            - aspect title: the title of the aspect that the two lists of text snippets are about
            - aspect detail: the description of the aspect that the two lists of text snippets are about
        - list 1: the first list of text snippets that are separated with new lines. Each snippet is a short text that is related to the aspect
        - list 2: the second list of text snippets that are separated with new lines. Each snippet is a short text that is related to the aspect
    
    ## Your output should be a valid JSON object in ```json``` block that contains the following keys:
        - reason: a string that shows the reason of the match or not match
        - is_matched: a boolean value that shows if the two lists of text snippets are matched. If they are matched, the value should be True, otherwise, it should be false.
    
    # aspect

    aspect title: {aspect_1['aspect']}
    aspect detail: {aspect_1['description']}

    # list 1
    {text_snippets_1}

    # list 2
    {text_snippets_2}

    # output: ```json
    """

def create_styling_matching_prompt(aspect_1, aspect_2):
    text_snippets_1 = "\n".join(aspect_1['evidence'])
    text_snippets_2 = "\n".join(aspect_2['evidence'])
    return f"""You are a helpful assistant and an expert judge with a keen eye for writing style. Your task is to carefully assess two lists of text snippets and determine whether they match in writing style. For each comparison, consider elements such as tone, vocabulary, sentence structure, and overall flow. Evaluate the degree of consistency, noting any significant differences or similarities in style. Provide a clear decision on whether the styles match or differ.

    # Your task:
    Given two lists of text snippets, you should determine if the two lists match in writing style. If the two lists are similar in terms of tone, vocabulary, sentence structure, and overall flow, then they are matched. Otherwise, they are not matched.

    ## Your inputs:
        - list 1: the first list of text snippets that are seperated with new lines. Each snippet is one or more than one sentences.
        - list 2: the second list of text snippets that are seperated with new lines. Each snippet is one or more than one sentences.
    
    ## Your output should be a valid JSON object in ```json``` block that contains the following keys:
        - reason: a string that shows the reason of the match or not match
        - is_matched: a boolean value that shows if the two lists of text snippets are matched in writing style. If they are matched, the value should be True, otherwise, it should be false.
    
    # list 1
    {text_snippets_1}

    # list 2
    {text_snippets_2}

    # output: ```json
    """

def get_match_table(expected_aspects, generated_aspects, llm, max_length_generation=8096, max_retries=100, ignore_on_fail=False):
    redo_examples = []
    temperature = 0.0
    attempt = 0
    matching_table = {key: {"recall": dict(), "precision": dict(), "generated_output_aspects": generated_aspects[key], "expected_output_aspects": expected_aspects[key]} for key in expected_aspects.keys()}
    key_to_aspect_to_name_gen = {key: {x['aspect']: x for x in aspects} for key, aspects in generated_aspects.items()}
    key_to_aspect_to_name_exp = {key: {x['aspect']: x for x in aspects} for key, aspects in expected_aspects.items()}
    for key in expected_aspects.keys():
        exp_aspects = expected_aspects[key]
        gen_aspects = generated_aspects[key]
        for i, exp_aspect in enumerate(exp_aspects):
            prompt = create_aspect_matching_prompt(exp_aspect, gen_aspects)
            redo_examples.append((key, "exp", exp_aspect['aspect'], prompt))
        for i, gen_aspect in enumerate(gen_aspects):
            prompt = create_aspect_matching_prompt(gen_aspect, exp_aspects)
            redo_examples.append((key, "gen", gen_aspect['aspect'], prompt))
    matched_to_check = []
    while len(redo_examples) > 0:
        prompts = [prompt for key, _, _, prompt in redo_examples]
        sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_length_generation, stop="```")
        outputs = llm.generate(prompts, sampling_params)
        new_redo_examples = []
        for (key, aspect_type, aspect_name, prompt), output in zip(redo_examples, outputs):
            try:
                match = parse_json_aspect_matching(output.outputs[0].text)
                if match["is_matched"]:
                    if aspect_type == "exp":
                        aspect_exp = key_to_aspect_to_name_exp[key][aspect_name]
                        aspect_gen = key_to_aspect_to_name_gen[key][match["matched_aspect"]]
                        matched_to_check.append((key, "exp", "content", aspect_exp, aspect_gen, create_content_matching_prompt(aspect_exp, aspect_gen)))
                        matched_to_check.append((key, "exp", "style", aspect_exp, aspect_gen, create_styling_matching_prompt(aspect_exp, aspect_gen)))
                        matching_table[key]["recall"][aspect_name] = {"matched": True, "details": {"matched_aspect": match["matched_aspect"], "is_concept_matched": True}}
                    else:
                        aspect_gen = key_to_aspect_to_name_gen[key][aspect_name]
                        aspect_exp = key_to_aspect_to_name_exp[key][match["matched_aspect"]]
                        matched_to_check.append((key, "gen", "content", aspect_gen, aspect_exp, create_content_matching_prompt(aspect_gen, aspect_exp)))
                        matched_to_check.append((key, "gen", "style", aspect_gen, aspect_exp, create_styling_matching_prompt(aspect_gen, aspect_exp)))
                        matching_table[key]["precision"][aspect_name] = {"matched": True, "details": {"matched_aspect": match["matched_aspect"], "is_concept_matched": True}}
                else:
                    if aspect_type == "exp":
                        matching_table[key]["recall"][aspect_name] = {"matched": False, "details": {}}
                    else:
                        matching_table[key]["precision"][aspect_name] = {"matched": False, "details": {}}
            except Exception as e:
                print(f"Error: {e}")
                new_redo_examples.append((key, aspect_type, aspect_name, prompt))
        redo_examples = new_redo_examples
        if temperature < 1.0:
            temperature += 0.1
        attempt += 1
        if attempt > max_retries:
            not_possible_generate = [(key, aspect_type, aspect_name) for key, aspect_type, aspect_name, prompt in redo_examples]
            print("Could not generate aspects for ids: " + str(not_possible_generate))
            if ignore_on_fail:
                for (key, aspect_type, aspect_name) in not_possible_generate:
                    if aspect_type == "exp":
                        matching_table[key]["recall"][aspect_name] = {"matched": False, "details": {}}
                    else:
                        matching_table[key]["precision"][aspect_name] = {"matched": False, "details": {}}
                break
            else:
                raise Exception("Could not generate aspects for ids: " + str(not_possible_generate))
    temperature = 0.0
    attempt = 0
    while len(matched_to_check) > 0:
        prompts = [prompt for key, aspect_type, _, _, _, prompt in matched_to_check]
        sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_length_generation, stop="```")
        outputs = llm.generate(prompts, sampling_params)
        new_matched_to_check = []
        for (key, aspect_type, similarity_mode, aspect_1, aspect_2, prompt), output in zip(matched_to_check, outputs):
            try:
                match = parse_json_content_style_matching(output.outputs[0].text)
                if aspect_type == "exp":
                    if similarity_mode == "content":
                        matching_table[key]["recall"][aspect_1['aspect']]['details']['is_content_matched'] = match["is_matched"]
                        matching_table[key]["recall"][aspect_1['aspect']]['details']['is_content_matched_reason'] = match["reason"]
                        
                    else:
                        matching_table[key]["recall"][aspect_1['aspect']]['details']['is_style_matched'] = match["is_matched"]
                        matching_table[key]["recall"][aspect_1['aspect']]['details']['is_style_matched_reason'] = match["reason"]
                else:
                    if similarity_mode == "content":
                        matching_table[key]["precision"][aspect_1['aspect']]['details']['is_content_matched'] = match["is_matched"]
                        matching_table[key]["precision"][aspect_1['aspect']]['details']['is_content_matched_reason'] = match["reason"]
                    else:
                        matching_table[key]["precision"][aspect_1['aspect']]['details']['is_style_matched'] = match["is_matched"]
                        matching_table[key]["precision"][aspect_1['aspect']]['details']['is_style_matched_reason'] = match["reason"]
            except Exception as e:
                print(f"Error: {e}")
                new_matched_to_check.append((key, aspect_type, similarity_mode, aspect_1, aspect_2, prompt))
        matched_to_check = new_matched_to_check
        if temperature < 1.0:
            temperature += 0.1
        attempt += 1
        if attempt > max_retries:
            not_possible_generate = [(key, aspect_type, aspect_type_2, aspect_1, aspect_2) for key, aspect_type, aspect_type_2, aspect_1, aspect_2, prompt in matched_to_check]
            print("Could not generate aspects for ids: " + str(not_possible_generate))
            if ignore_on_fail:
                for (key, aspect_type, aspect_type_2, aspect_1, aspect_2) in not_possible_generate:
                    if aspect_type == "exp":
                        matching_table[key]["recall"][aspect_1['aspect']]['is_matched'] = False
                        matching_table[key]["recall"][aspect_1['aspect']]['details']['is_content_matched'] = False
                        matching_table[key]["recall"][aspect_1['aspect']]['details']['is_content_matched_reason'] = "Could not generate the output"
                        matching_table[key]["recall"][aspect_1['aspect']]['details']['is_style_matched'] = False
                        matching_table[key]["recall"][aspect_1['aspect']]['details']['is_style_matched_reason'] = "Could not generate the output"
                    else:
                        matching_table[key]["precision"][aspect_1['aspect']]['is_matched'] = False
                        matching_table[key]["precision"][aspect_1['aspect']]['details']['is_content_matched'] = False
                        matching_table[key]["precision"][aspect_1['aspect']]['details']['is_content_matched_reason'] = "Could not generate the output"
                        matching_table[key]["precision"][aspect_1['aspect']]['details']['is_style_matched'] = False
                        matching_table[key]["precision"][aspect_1['aspect']]['details']['is_style_matched_reason'] = "Could not generate the output"
                break
            else:
                raise Exception("Could not generate aspects for ids: " + str(not_possible_generate))    
    return matching_table