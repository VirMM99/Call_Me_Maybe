"""Main entry point for Call Me Maybe function calling system."""

import argparse
import json
import os
import sys
from typing import Any
from llm_sdk import Small_LLM_Model
from .file_loader import load_fn_definitions, load_prompts, ParsingFileError
from .basemodels import FunctionDefinitionCheck, PromptItem, FunctionCallOutput
from .constrained_decode import constrained_decode_function_name, constrained_decode_parameters
from .validator import validate_function_call
import time


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Function calling system using constrained decoding"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to function definitions JSON file"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON file"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to output results JSON file"
    )
    return parser.parse_args()


def extract_parameters_from_json(
        json_text: str,
        function_def: FunctionDefinitionCheck
        ) -> dict[str, Any]:
    """Parse generated JSON text and extract parameters dict."""
    try:
        parsed = json.loads(json_text)
        params: dict[str, Any] = {}
        for name, param in function_def.parameters.items():
            value = parsed[name]
            if param.type.value == "number":
                value = float(value)
            elif param.type.value == "integer":
                value = int(value)
            params[name] = value
        return params
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to extract parameters from JSON: {e}")


def normalize_string_value(value: str, prompt: str, fn_name: str) -> str:
    """remove a leaked function name and prefer the prompt's exact casting"""
    if value.endswith(fn_name):
        value = value[: -len(fn_name)]
    low = value.lower()
    idx = prompt.lower().find(low)
    if idx != -1:
        value = prompt[idx: idx + len(value)]
    return value


def build_prompt(
        prompt: str,
        function_definitions: list[FunctionDefinitionCheck]
) -> str:
    """Build the model prompt, listing the available functions."""
    lines: list[str] = ["Available functions:"]
    for func in function_definitions:
        params = ",".join(
            f"{name} ({param.type.value})"
            for name, param in func.parameters.items()
        )
        lines.append(f"- {func.name}({params}): {func.description}")
    lines.append(f"Question: {prompt}")
    lines.append("Function:")
    return "\n".join(lines)


def build_prompt_param(
        prompt: str,
        fn_name: str,
        fn_description: str,
        fn_parameters: dict
) -> str:
    """This create the estructured context text
    for the generation phase of parameters
    """
    # We format the params of the function so LLM knows what is coming
    params_info = ", ".join([
        f"{name}: {param.type.value}"
        for name, param in fn_parameters.items()
    ]) 

    # Clear structure that guides the model to 
    # the generation of the JSON of arguments
    return (
        f"Using this function"
        f"{fn_name}({params_info})\n"
        f"Function description: {fn_description}\n"
        f"Provide the arguments to solve user prompt in JSON format:\n"
        f'[\n{{\n"prompt": "{prompt}",\n'
        f'"name": "{fn_name}",\n"parameters":\n{{'
    )

def generate_function_call(
        prompt: str,
        llm_model: Small_LLM_Model,
        list_fn_name: list[str],
        function_definitions: list[FunctionDefinitionCheck]
        ) -> tuple[str, dict[str, Any]]:
    """Generate function name and parameters for a single prompt."""
    input_ids: list[int] = ( 
                llm_model.encode(build_prompt(prompt, function_definitions)
                ).tolist()[0]
            )
    generate_ids: list[int] = []
    selected_function: str | None = None

    # PRINT DE DEBUGEO: CONTEXTO INICIAL PARA BUSCAR EL NOMBRE DE LA FUNCION:
    print("\n//////// Contexto inicial para encontrar FN_NAME:")
    for i in input_ids:
        print(llm_model.decode([i]), end="")
    # fin print
    # this for is to find the name of the function with a specific function
    for _ in range(15):
        logits: list[float] = llm_model.get_logits_from_input_ids(input_ids)
        next_token, _, _ = constrained_decode_function_name(
            logits,
            list_fn_name,
            llm_model,
            generate_ids
        )

        if next_token == 128247:
            selected_function = llm_model.decode(generate_ids)
            break

        input_ids.append(next_token)
        generate_ids.append(next_token)

    else:
        raise ValueError("Function selection did not complete")

    if selected_function is None:
        raise ValueError("No function selected")

    selected_definition = None
    for func in function_definitions:
        if func.name == selected_function:
            selected_definition = func
            break

    if selected_definition is None:
        raise ValueError(
            f"Function definition not found: {selected_function}"
            )

    # We use build_promt_param to find the context of params
    prompt_param_text = build_prompt_param(
        prompt,
        selected_definition.name,
        selected_definition.description,
        selected_definition.parameters
    )

    input_ids = llm_model.encode(prompt_param_text).tolist()[0]

    generate_ids = []
    param_index = 0
    arg_state = "start"
    for _ in range(80):
        logits = llm_model.get_logits_from_input_ids(input_ids)
        # print debugueo
        print("\n//////// Contexto para encontrar parametros (token a token):")
        for i in input_ids:
            print(llm_model.decode([i]), end="")
        #Fin del print
        next_token, param_index, arg_state = constrained_decode_parameters(
            logits,
            function_definitions,
            llm_model,
            generate_ids,
            selected_function,
            param_index,
            arg_state
        )

        input_ids.append(next_token)
        generate_ids.append(next_token)

        if arg_state == "end":
            break
    else:
        raise ValueError("Argument generation did not complete")

    decoded = llm_model.decode(generate_ids)

    # This two whiles clean extra char.
    # Now LLM can find and select tokens that I have
    # banned before
    while decoded.endswith("}}"):
        decoded = decoded[:-1]
    while decoded.endswith(","):
            decoded = decoded[:-1] + '}'
    
    parameters = extract_parameters_from_json(
            decoded, selected_definition
        )
    
    return selected_function, parameters


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        loaded_functions: list[FunctionDefinitionCheck] = load_fn_definitions(
            args.functions_definition
        )
        test_prompts: list[PromptItem] = load_prompts(args.input)
    except ParsingFileError as e:
        print(f"Error loading input files: {e}", file=sys.stderr)
        return 1

    if not loaded_functions:
        print("No valid function definitions loaded", file=sys.stderr)
        return 1

    if not test_prompts:
        print("No valid prompts loaded", file=sys.stderr)
        return 1

    llm_model: Small_LLM_Model = Small_LLM_Model()
    list_fn_name: list[str] = [fn.name for fn in loaded_functions]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results: list[dict[str, Any]] = []

    init_time: float = time.perf_counter()

    for test in test_prompts:
        prompt = test.prompt
        try:
            fn_name, parameters = generate_function_call(
                prompt,
                llm_model,
                list_fn_name,
                loaded_functions
            )

            result = {
                "prompt": prompt,
                "name": fn_name,
                "parameters": parameters
            }
            results.append(result)

            selected_def = next(
                f for f in loaded_functions if f.name == fn_name
            )
            validate_function_call(
                FunctionCallOutput(
                    prompt=prompt,
                    name=fn_name,
                    parameters=parameters
                ),
                selected_def
            )
            print(f"OK: {prompt} -> {fn_name}({parameters})")

        except Exception as e:
            print(f"Error processing '{prompt}': {e}", file=sys.stderr)
            return 1

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")
    except OSError as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        return 1
    final_program_time: float = time.perf_counter()
    final_duration = final_program_time - init_time
    print(
        f"Total time processing all prompts: "
        f"{final_duration:.2f} seconds"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("YOU ARE THE PROBLEM")
