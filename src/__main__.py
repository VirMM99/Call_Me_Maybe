"""Main entry point for Call Me Maybe function calling system."""

import argparse
import json
import os
import sys
from typing import Any
from llm_sdk import Small_LLM_Model
from .file_loader import load_fn_definitions, load_prompts, ParsingFileError
from .basemodels import FunctionDefinitionCheck, PromptItem, FunctionCallOutput
from .constrained_decode import constrained_decode_fn
from .validator import validate_function_call


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
        return {
            name: parsed[name]
            for name in function_def.parameters.keys()
        }
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to extract parameters from JSON: {e}")


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

    for _ in range(50):
        logits: list[float] = llm_model.get_logits_from_input_ids(input_ids)
        next_token, _, _ = constrained_decode_fn(
            logits,
            list_fn_name,
            function_definitions,
            llm_model,
            generate_ids,
            selected_function
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

    generate_ids = []
    param_index = 0
    arg_state = "start"
    for _ in range(200):
        logits = llm_model.get_logits_from_input_ids(input_ids)
        next_token, param_index, arg_state = constrained_decode_fn(
            logits,
            list_fn_name,
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
