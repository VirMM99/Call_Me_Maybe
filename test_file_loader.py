from src.file_loader import load_fn_definitions, load_prompts


def main() -> None:
    """Test the JSON file loaders."""
    functions = load_fn_definitions(
        "data/input/functions_definition.json"
    )
    prompts = load_prompts(
        "data/input/function_calling_tests.json"
    )

    print("FUNCTIONS:")
    for function in functions:
        print(function)
        print()

    print("PROMPTS:")
    for prompt in prompts:
        print(prompt)
        print()


if __name__ == "__main__":
    main()
