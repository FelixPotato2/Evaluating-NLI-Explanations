from google import genai
import os
import evaluate_LLMs as ev
import prompts_construction as pc
import time
import re, json
import ast

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
    if k not in merged:
        merged[k] = v
    else:
        if isinstance(merged[k], list):
            merged[k].append(v)
        else:
            merged[k] = [merged[k], v]

def _merge_into(merged: dict, obj) -> int:
    ingested = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            _append_value(merged, str(k).lower(), v)
            ingested += 1
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                for k, v in item.items():
                    _append_value(merged, str(k).lower(), v)
                    ingested += 1
    return ingested

def _extract_balanced_json_regions(text: str):
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
                        break
            i += 1

        if not stack:
            regions.append(text[start:i])

    return regions

_qid_pat = re.compile(r'"(q\d{4})"\s*:', re.IGNORECASE)

def extract_qids_from_text(raw: str):
    seen = set()
    out = []
    for m in _qid_pat.finditer(raw):
        qid = m.group(1).lower()
        if qid not in seen:
            seen.add(qid)
            out.append(qid)
    return out

# NEW: parse lines like: q0617:{'answer': 'entailment', 'explanation': [...]}
_line_kv_pat = re.compile(r"^\s*(q\d{4})\s*:\s*(\{.*\})\s*$", re.IGNORECASE)

def _parse_qid_python_kv_lines(text: str) -> dict:
    """
    Extracts and parses lines of the form:
      q0123:{'answer': '...', 'explanation': [...]}
      q0123:{"answer": "...", "explanation": [...]}
    Uses ast.literal_eval for single-quote dicts, json.loads for JSON dicts.
    Returns dict: {qid: parsed_value_dict, ...}
    """
    out = {}
    for line in text.splitlines():
        m = _line_kv_pat.match(line)
        if not m:
            continue
        qid = m.group(1).lower()
        payload = m.group(2).strip()

        # Try JSON first (handles the q0694:{"answer":...} lines)
        try:
            val = json.loads(_strip_trailing_commas(payload))
            if isinstance(val, dict):
                out[qid] = val
                continue
        except Exception:
            pass

        # Fallback: Python literal dict (single quotes)
        try:
            val = ast.literal_eval(payload)
            if isinstance(val, dict):
                out[qid] = val
        except Exception:
            continue

    return out

def parse_llm_json_mixed(text: str):
    fence_pat = re.compile(r"```+\s*(?:json)?\s*(.*?)\s*```+",
                           flags=re.IGNORECASE | re.DOTALL)
    fenced_blocks = fence_pat.findall(text)

    merged = {}
    bad_blocks = 0
    merged_items = 0
    bad_qids = []

    def _try_ingest(raw: str):
        nonlocal bad_blocks, merged_items, bad_qids
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
            bad_qids.extend(extract_qids_from_text(raw))
            return False

    # 1) fenced JSON blocks
    for raw in fenced_blocks:
        _try_ingest(raw)

    # 2) non-fenced balanced JSON regions
    text_wo_fences = fence_pat.sub("", text)
    for region in _extract_balanced_json_regions(text_wo_fences):
        _try_ingest(region)

    # 3) NEW: line-by-line q####:{...} (python/JSON dict payload)
    kv = _parse_qid_python_kv_lines(text_wo_fences)
    for k, v in kv.items():
        _append_value(merged, k, v)
        merged_items += 1

    # unique, preserve order
    seen = set()
    bad_qids_unique = []
    for q in bad_qids:
        if q not in seen:
            seen.add(q)
            bad_qids_unique.append(q)

    return merged, len(fenced_blocks), bad_blocks, merged_items, bad_qids_unique


def select_examples_by_qids(examples, qids):
    """
    Returns the FULL example objects corresponding to qids.

    Supports:
      - examples is a dict: {"q0001": example_obj, ...}
      - examples is a list of dicts: [{"id":"q0001", ...}, ...]
      - examples is a list of strings: ["q0001:{...}", ...]   <-- NEW
    """
    if not qids:
        return []

    qset = set(qids)

    if isinstance(examples, dict):
        return [examples[q] for q in qids if q in examples]

    if isinstance(examples, list):
        # list of dicts with "id"
        if examples and isinstance(examples[0], dict):
            idx = {}
            for ex in examples:
                if isinstance(ex, dict) and "id" in ex:
                    idx[str(ex["id"]).lower()] = ex
            return [idx[q] for q in qids if q in idx]

        # NEW: list of strings "q####:...."
        out = []
        for s in examples:
            if not isinstance(s, str):
                continue
            m = re.match(r"^\s*(q\d{4})\s*:", s, flags=re.IGNORECASE)
            if m and m.group(1).lower() in qset:
                out.append(s)
        return out

    raise TypeError(f"Unsupported examples type: {type(examples)}")


def generate(fixed_prompt, examples, amount, client, generated_file, model,
             delay=0.8, encoding="utf-8"):

    with open(fixed_prompt + '.txt', "r", encoding=encoding) as file:
        fixed = file.read()

    # allow model to be str or list[str]
    models = model if isinstance(model, list) else [model]

    paths_by_model = {}

    for model_name in models:
        # model-specific output txt/json
        out_txt = f"{generated_file}_{model_name}.txt"
        open(out_txt, "w", encoding=encoding).close()

        prompts = pc.construct_prompt(fixed, examples, amount)

        # PASS 1
        for i, prompt in enumerate(prompts):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                with open(out_txt, "a", encoding=encoding) as resp:
                    resp.write(response.text)
                    resp.write("\n")
            except Exception as e:
                print("Error at", i, e)
                time.sleep(10)
                continue
            time.sleep(delay)

        with open(out_txt, "r", encoding=encoding) as f:
            text = f.read()

        merged, nblocks, bad_blocks, merged_items, bad_qids = parse_llm_json_mixed(text)

        if not merged:
            raise ValueError(f"[{model_name}] No valid JSON blocks were found in the model output.")

        # PASS 2: rerun MISSING qids (more reliable than parsing malformed block)
        expected_qids = [f"q{i:04d}" for i in range(1, amount + 1)]
        missing_qids = [q for q in expected_qids if q not in merged]

        if missing_qids:
            missing_examples = select_examples_by_qids(examples, missing_qids)
            if missing_examples:
                rerun_prompts = pc.construct_prompt(fixed, missing_examples, len(missing_examples))
                for i, prompt in enumerate(rerun_prompts):
                    try:
                        response = client.models.generate_content(model=model_name, contents=prompt)
                        with open(out_txt, "a", encoding=encoding) as resp:
                            resp.write(response.text)
                            resp.write("\n")
                    except Exception as e:
                        print("Rerun error at", i, e)
                        time.sleep(10)
                        continue
                    time.sleep(delay)

                with open(out_txt, "r", encoding=encoding) as f:
                    text2 = f.read()
                merged, nblocks2, bad_blocks2, merged_items2, bad_qids2 = parse_llm_json_mixed(text2)

        out_json = f"final_{generated_file}_{model_name}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"[{model_name}] merged {merged_items} items, skipped {bad_blocks} malformed blocks")
        if missing_qids:
            print(f"[{model_name}] missing after pass-1: {len(missing_qids)} (rerun attempted: {len(missing_examples)})")

        paths_by_model[model_name] = out_json

    # keep old behavior if caller passed a single model string
    if isinstance(model, list):
        return paths_by_model
    return paths_by_model[models[0]]