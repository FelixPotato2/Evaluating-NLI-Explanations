from google import genai
import os
import evaluate_LLMs as ev
import prompts_construction as pc
import time
import re, json

def _strip_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)

def _maybe_fix_bracketed_object(s: str) -> str:
    ss = s.strip()
    if ss.startswith("[") and ss.endswith("]"):
        inner = ss[1:-1].strip()
        if re.search(r'"\s*[^"]+\s*"\s*:', inner):
            return "{" + inner + "}"
    return s

def _append_value(merged: dict, k: str, v):
    """Preserve duplicates: store multiple values under same key as a list."""
    if k not in merged:
        merged[k] = v
    else:
        if isinstance(merged[k], list):
            merged[k].append(v)
        else:
            merged[k] = [merged[k], v]

def _merge_into(merged: dict, obj) -> int:
    """
    Merge parsed JSON into merged dict.
    Preserves duplicate ids by storing values as a list under that id.
    Returns number of id->value pairs ingested.
    """
    ingested = 0

    if isinstance(obj, dict):
        for k, v in obj.items():
            _append_value(merged, k, v)
            ingested += 1

    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                # list of {id: {...}} dicts
                for k, v in item.items():
                    _append_value(merged, k, v)
                    ingested += 1

    return ingested

def _extract_balanced_json_regions(text: str):
    """
    Extract balanced top-level JSON objects/arrays from raw text.
    Ignores brackets/braces inside JSON strings.

    Returns list of substrings that start with '{' or '[' and end at matching '}' or ']'.
    """
    regions = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue

        start = i
        stack = [ch]
        i += 1

        in_str = False
        escape = False

        while i < n and stack:
            c = text[i]

            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c in "{[":
                    stack.append(c)
                elif c in "}]":
                    opener = stack[-1]
                    if (opener == "{" and c == "}") or (opener == "[" and c == "]"):
                        stack.pop()
                    else:
                        # mismatched; abort this region
                        break
            i += 1

        if not stack:
            regions.append(text[start:i])
        # else: incomplete/mismatched; ignore

    return regions

def parse_llm_json_mixed(text: str):
    """
    Parses:
      - fenced ```json ... ``` blocks
      - unfenced standalone JSON objects/arrays in the text
    Returns: merged, n_fenced_blocks, bad_blocks, merged_items
    """

    # 1) Pull fenced blocks first (only those explicitly marked json).
    fence_pat = re.compile(r"```+\s*(?:json)?\s*(.*?)\s*```+", flags=re.IGNORECASE | re.DOTALL)
    fenced_blocks = fence_pat.findall(text)

    merged = {}
    bad_blocks = 0
    merged_items = 0

    # Helper to normalize + parse + merge
    def _try_ingest(raw: str):
        nonlocal bad_blocks, merged_items
        candidate = raw.strip()
        candidate = _strip_trailing_commas(candidate)
        candidate = _maybe_fix_bracketed_object(candidate)
        candidate = _strip_trailing_commas(candidate)
        try:
            obj = json.loads(candidate)
            merged_items += _merge_into(merged, obj)
            return True
        except json.JSONDecodeError:
            bad_blocks += 1
            return False

    # ingest fenced
    for raw in fenced_blocks:
        _try_ingest(raw)

    # 2) Extract unfenced JSON regions too (covers your bare [...] lists).
    # To avoid double counting, remove fenced blocks from consideration by blanking them out.
    text_wo_fences = fence_pat.sub("", text)

    for region in _extract_balanced_json_regions(text_wo_fences):
        # try to ingest; if it fails it's probably not JSON but some brace text
        _try_ingest(region)

    return merged, len(fenced_blocks), bad_blocks, merged_items

def generate(fixed_prompt, examples, amount, client, generated_file, model, delay=0.8, encoding="utf-8"):

    with open(fixed_prompt + '.txt', "r", encoding=encoding) as file:
        fixed = file.read()

    prompts = pc.construct_prompt(fixed, examples, amount)
    open(generated_file + '.txt', "w", encoding=encoding).close()

    for i, prompt in enumerate(prompts):
        try:
            response = client.models.generate_content(   
                model=model,
                contents=prompt
            )
            with open(generated_file + '.txt', "a", encoding=encoding) as resp:  
                resp.write(response.text)
                resp.write("\n")  
        except Exception as e:
            print("Error at", i, e)
            time.sleep(10)
            continue
        time.sleep(delay)

    with open(generated_file + ".txt", "r", encoding=encoding) as f:
        text = f.read()

    merged, nblocks, bad_blocks, merged_items = parse_llm_json_mixed(text)

    if not merged:
        raise ValueError("No valid JSON blocks were found in the model output.")

    with open(f"final_{generated_file}.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Found {nblocks} fenced blocks; skipped {bad_blocks} malformed blocks; merged {merged_items} items.")
    print(f"Unique ids in merged output: {len(merged)}")
    total_preds = sum(len(v) if isinstance(v, list) else 1 for v in merged.values())
    print(f"Total predictions (including duplicates): {total_preds}")
    return f"final_{generated_file}.json"

def count_pairIDs(json_path, *, verbose=True):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level dict, got {type(data)}")

    unique_ids = len(data)
    total_predictions = sum(len(v) if isinstance(v, list) else 1 for v in data.values())

    if verbose:
        print(f"Problems in JSON (unique ids): {unique_ids}")
        print(f"Problems in JSON (including duplicates): {total_predictions}")

    return unique_ids, total_predictions