from src.file_loader import ParsingFileError, load_fn_definitions


def main() -> None:
    """Test file loader error handling."""
    try:
        functions = load_fn_definitions("bad_item.json")
        print("VALID FUNCTIONS:")
        for function in functions:
            print(function)
    except ParsingFileError as error:
        print(f"Caught expected error: {error}")


if __name__ == "__main__":
    main()
