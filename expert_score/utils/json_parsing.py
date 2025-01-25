import json
import json5
from dataclasses import dataclass
import re
from typing import List
import copy
import parse

def _custom_json_parser_for_aspects(input):
    @dataclass
    class ParsedOutput:
        aspect: str
        description: str
        evidence: List[str]

    def remove_leading_trailing_quotes(input):
        if input.startswith("\""):
            input = input[1:]
        if input.endswith("\""):
            input = input[:-1]
        return input

    def parse_evidence_list(input):
        split_input = input.split('",')
        split_input = [
            remove_leading_trailing_quotes(s)
            .replace("\"", '"')
            .replace("\'", "'")
            for s in split_input
        ]
        return split_input

    x = copy.deepcopy(input)
    regex_newline_whitespace = re.compile(r"\n\s+")
    x = regex_newline_whitespace.sub(r"", x).strip()
    x = r" ".join(x.split())
    x = x.replace(": ", ":")
    split = x.split("},{")
    split[0] = split[0].lstrip("[{")
    split[-1] = split[-1][: split[-1].rfind("]") - 1]
    parsed_outputs = []
    total, could_not_parse = 0, 0
    for s in split:
        total += 1
        output = parse.parse(
            "\"aspect\":\"{}\",\"description\":\"{}\",\"evidence\":[{}]", s
        )
        if output is None:
            output = parse.parse(
               "\"aspect\":\"{}\",\"evidence\":[{}],\"description\":\"{}\"", s
            )
        if output is None:
            output = parse.parse(
               "\"evidence\":[{}],\"aspect\":\"{}\",\"description\":\"{}\"", s
            )
        if output is None:
            output = parse.parse(
               "\"evidence\":[{}],\"description\":\"{}\",\"aspect\":\"{}\"", s
            )
        if output is None:
            output = parse.parse(
               "\"description\":\"{}\",\"evidence\":[{}],\"aspect\":\"{}\"", s
            )
        if output is None:
            output = parse.parse(
               "\"description\":\"{}\",\"aspect\":\"{}\",\"evidence\":[{}]", s
            )
        if output is not None:
            parsed_outputs.append(
                ParsedOutput(
                    output[0], output[1], parse_evidence_list(output[2])
                )
            )
    if len(parsed_outputs) == 0:
        raise Exception("Could not parse the input")    
    return [{"aspect": p.aspect, "description": p.description, "evidence": p.evidence} for p in parsed_outputs]

def _custom_json_parser_for_matching(input):
    pass

def _custom_json_parser_for_content_style_matching(input):
    pass

def parse_json_aspect_matching(json_str):
    json_str = json_str.replace("```json", "").replace("```", "")
    try:
        return json.loads(json_str, strict=False)
    except Exception as e:
        pass
    try:
        return json5.loads(json_str)
    except Exception as e:
        pass
    try:
        return _custom_json_parser_for_matching(json_str)
    except Exception as e:
        raise Exception("Could not parse the input")

def parse_json_aspects(json_str):
    json_str = json_str.replace("```json", "").replace("```", "")
    try:
        return json.loads(json_str, strict=False)
    except Exception as e:
        pass
    try:
        return json5.loads(json_str)
    except Exception as e:
        pass
    try:
        return _custom_json_parser_for_aspects(json_str)
    except Exception as e:
        raise Exception("Could not parse the input")

def parse_json_content_style_matching(json_str):
    json_str = json_str.replace("```json", "").replace("```", "")
    try:
        return json.loads(json_str, strict=False)
    except Exception as e:
        pass
    try:
        return json5.loads(json_str)
    except Exception as e:
        pass
    try:
        return _custom_json_parser_for_content_style_matching(json_str)
    except Exception as e:
        raise Exception("Could not parse the input")