# Call Me Maybe - Agent Reference

## Project Goal
Build a function calling system using **constrained decoding** with Qwen/Qwen3-0.6B to translate natural language prompts into structured function calls with **100% valid JSON output**.

---

## Critical Rules (MUST FOLLOW)

### Input/Output
- **Input files**: `data/input/functions_definition.json`, `data/input/function_calling_tests.json`
- **Output file**: `data/output/function_calling_results.json` (or custom via `--output`)
- **CLI**: `uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]`

### Implementation Requirements
- **Python 3.10+**, flake8, mypy (strict recommended)
- **Pydantic** for ALL class validation
- **uv** for dependency management (`uv sync` must work)
- **No private SDK methods/attributes** from `llm_sdk`
- **Constrained decoding mandatory** - NOT prompting-based
- Handle all errors gracefully, never crash

### Forbidden
- `dspy`, `pytorch`, `huggingface`, `transformers`, `outlines` (except via provided `llm_sdk`)
- Hardcoding solutions for specific test cases

---

## Function Definitions (from functions_definition.json)

| Function | Parameters | Returns |
|----------|------------|---------|
| `fn_add_numbers` | `a: number`, `b: number` | `number` |
| `fn_greet` | `name: string` | `string` |
| `fn_reverse_string` | `s: string` | `string` |
| `fn_get_square_root` | `a: number` | `number` |
| `fn_substitute_string_with_regex` | `source_string: string`, `regex: string`, `replacement: string` | `string` |

**Private functions** (moulinette only): `fn_multiply_numbers`, `fn_is_even`, `fn_calculate_compound_interest`, `fn_execute_sql_query`, `fn_read_file`, `fn_format_template`

---

## Output Format (REQUIRED)

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

### Validation Rules
- Valid JSON (no trailing commas, no comments)
- Exact keys: `prompt`, `name`, `parameters`
- All required arguments present
- Types match function definition (`number` accepts int/float)
- No extra keys or prose

---

## Constrained Decoding Approach

### Core Concept
At each generation step:
1. Model produces logits for all tokens
2. Identify valid tokens that maintain **valid JSON structure** + **schema compliance**
3. Set invalid token logits to `-inf`
4. Select from remaining valid tokens

### Two Phases
1. **Function Selection**: Generate function name token-by-token, allow EOS (128247) when complete
2. **Argument Generation**: Generate JSON arguments per schema (handle multiple params)

### Vocabulary Usage
- Use `get_path_to_vocab_file()` to load `vocab.json`
- Map tokens ↔ token IDs for constraint checking
- Token strings may include leading spaces (e.g., `" 2"` vs `"2"`)

---

## Performance Targets
- **>90% accuracy** on function selection + argument extraction
- **100% valid JSON** (parseable)
- **<5 minutes** for all test prompts
- Reliable across multiple runs

---

## LLM SDK Methods (llm_sdk.Small_LLM_Model)
- `encode(text: str) -> Tensor` - Tokenize text
- `decode(token_ids: List[int]) -> str` - Detokenize
- `get_logits_from_input_ids(input_ids: List[int]) -> List[float]` - Get next-token logits
- `get_path_to_vocab_file() -> str` - Path to vocab.json

---

## Makefile Targets (Required)
- `install` - Install deps
- `run` - Execute main script
- `debug` - Run with pdb
- `clean` - Remove caches
- `lint` - `flake8 . && mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`
- `lint-strict` (optional) - `flake8 . && mypy . --strict`

---

## Moulinette Grading
```bash
cd moulinette && uv sync
uv run python -m moulinette prepare_exercises --set private
uv run python -m moulinette grade_student_answers --set private --student_answer_path <path>
```

---

## Key Files to Modify
- `src/__main__.py` - Add CLI args, output writing, full pipeline
- `src/constrained_decode.py` - Fix multi-param handling, complete argument generation
- `src/validator.py` - Already complete
- `src/file_loader.py` - Already complete
- `src/basemodels.py` - Already complete

---

## Common Pitfalls to Avoid
1. Single-parameter assumption in `constrained_decode.py` (line 138: `next(iter(...))`)
2. Not handling string escaping in JSON generation
3. Not writing output file
4. Missing CLI argument parsing
5. Crashing on invalid input instead of graceful error