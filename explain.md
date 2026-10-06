# Call Me Maybe - Step-by-Step Explanation

## Overview
This project implements a **function calling system** using **constrained decoding** with a small LLM (Qwen/Qwen3-0.6B). The system translates natural language prompts into structured function calls with guaranteed valid JSON output.

---

## Step 1: Understand the Input Files

### `data/input/functions_definition.json`
Contains the available functions the system can call. Each function has:
- **name**: Function identifier (e.g., `fn_add_numbers`)
- **description**: Human-readable description
- **parameters**: Dictionary of parameter names with their types (`string`, `number`, `integer`)
- **returns**: Return type

Example:
```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {
    "a": {"type": "number"},
    "b": {"type": "number"}
  },
  "returns": {"type": "number"}
}
```

### `data/input/function_calling_tests.json`
Contains an array of natural language prompts to process:
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Reverse the string 'hello'"}
]
```

---

## Step 2: Load and Validate Input

**Files**: `src/file_loader.py`, `src/basemodels.py`

1. Read both JSON files
2. Validate against Pydantic models:
   - `FunctionDefinitionCheck` - validates function structure
   - `PromptItem` - validates prompt structure
3. Handle errors gracefully (missing files, invalid JSON, wrong structure)

---

## Step 3: Initialize the LLM

**File**: `llm_sdk/llm_sdk/__init__.py` → `Small_LLM_Model`

```python
llm_model = Small_LLM_Model()  # Uses Qwen/Qwen3-0.6B by default
```

Key methods:
- `encode(text)` → Tensor of token IDs
- `decode(token_ids)` → String
- `get_logits_from_input_ids(input_ids)` → List of logits for next token
- `get_path_to_vocab_file()` → Path to vocab.json

---

## Step 4: Constrained Decoding Pipeline

### Phase 1: Function Selection

**Goal**: Generate the function name token-by-token using constrained decoding.

**Process**:
1. Encode the prompt into token IDs
2. At each step, get logits from LLM
3. Determine valid next tokens:
   - If no tokens generated yet: allow first token of any function name
   - If prefix matches a function name: allow next token of that function
   - If complete function name generated: allow EOS token (128247)
4. Mask invalid tokens (set logits to `-inf`)
5. Select token with highest logit
6. Repeat until EOS token selected

**Key insight**: Use the vocabulary to map between function name strings and their token IDs.

### Phase 2: Argument Generation

**Goal**: Generate valid JSON arguments matching the function's schema.

**Process**:
1. Get the selected function's parameter definitions
2. Generate JSON object token-by-token: `{"param1": value1, "param2": value2, ...}`
3. At each step, determine valid tokens based on:
   - Current position in JSON structure (keys, colons, quotes, values, commas, braces)
   - Parameter type constraints (numbers only allow digits, strings allow quoted content)
   - Schema compliance (all required params present, correct types)

**Critical**: Handle multiple parameters (not just single param as in skeleton code).

---

## Step 5: Build Output

For each prompt, create an output object:
```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

Collect all results into an array and write to output file.

---

## Step 6: CLI Interface

**File**: `src/__main__.py`

Accept arguments:
- `--functions_definition` (default: `data/input/functions_definition.json`)
- `--input` (default: `data/input/function_calling_tests.json`)
- `--output` (default: `data/output/function_calling_results.json`)

---

## Step 7: Validation (Optional but Recommended)

Use `src/validator.py` to validate generated calls against definitions:
- Function exists
- All required parameters present
- Parameter types match definition

---

## Key Technical Details

### Vocabulary Mapping
```python
vocab_path = llm_model.get_path_to_vocab_file()
with open(vocab_path) as f:
    vocab = json.load(f)  # {token_string: token_id, ...}
```

### Token Constraints
- **Function names**: Exact token sequences from vocabulary
- **Numbers**: Tokens starting with digits
- **Strings**: Tokens without quotes or newlines (inside quotes)
- **JSON structure**: `{`, `}`, `"`, `:`, `,` tokens
- **EOS**: Token ID 128247

### Logit Masking
```python
for token_id in range(len(logits)):
    if token_id not in allowed_token_ids:
        logits[token_id] = float("-inf")
```

---

## Project Structure

```
src/
├── __main__.py           # Entry point, CLI, pipeline orchestration
├── basemodels.py         # Pydantic models for validation
├── file_loader.py        # Load & validate input JSON
├── validator.py          # Validate function calls
├── constrained_decode.py # Core constrained decoding logic
└── __init__.py
```

---

## Testing & Grading

### Local Testing
```bash
uv run python -m src
```

### Moulinette Grading
```bash
cd moulinette && uv sync
uv run python -m moulinette prepare_exercises --set private
uv run python -m moulinette grade_student_answers --set private --student_answer_path ../data/output/function_calling_results.json
```

---

## Common Challenges

1. **Multi-parameter functions**: Don't assume single parameter (fix line 138 in constrained_decode.py)
2. **String escaping**: Handle quotes, backslashes in string values
3. **Token space handling**: Tokenizer may include leading spaces (e.g., `" 2"` vs `"2"`)
4. **Number formats**: Handle integers, floats, negative numbers
5. **Performance**: Must complete in <5 minutes for all prompts

---

## Success Criteria

- ✅ 100% valid JSON output (parseable)
- ✅ >90% function selection accuracy
- ✅ >90% argument extraction accuracy
- ✅ <5 minutes execution time
- ✅ Graceful error handling
- ✅ CLI with proper arguments
- ✅ Output file written correctly