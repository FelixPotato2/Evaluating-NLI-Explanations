from google import genai
import os
import time
import re
import json
import ast
import prompts_construction as pc

_qid_key_pat = re.compile(
    r'(?:"(?P<dq>q\d{4})"|\'(?P<sq>q\d{4})\'|(?P<uq>\bq\d{4}\b))\s*:\s*',
    re.IGNORECASE
)

def _strip_trailing_commas(s):
    """
    Function to remove trailing commas before } or ]
    param: s: string of text that we want to analyse
    returns: string without } or ] if found
    """
    return re.sub(r",\s*([}\]])", r"\1", s)

def _normalize_qid(q):
    """
    Normalize the ids
    param: q: string
    returns: normalized string
    """
    return (q or "").strip().lower()

def _is_qid_key(k):
    """
    Check if qid is of the correct format, i.e. qxxxx
    param: k: string
    returns: bool
    """
    return bool(re.fullmatch(r"q\d{4}", (k or "").strip().lower()))

def _coerce_explanation(v):
    """
    Normalize explanation field into a flat list of strings.

    param: v (str, list, or None): Raw explanation value from LLM output.
    returns: list: Flattened list of explanation strings.
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        out = []
        for item in v:
            if item is None:
                continue
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, list):
                for x in item:
                    if isinstance(x, str):
                        out.append(x)
            else:
                continue
        return out
    return []

def _normalize_payload(obj):
    """
    Parse a string into a dictionary using json.loads or ast.literal_eval.

    param: s (str): String representation of a dictionary.
    returns: obj (dict or None): Parsed dictionary if successful, else None.
    """
    if not isinstance(obj, dict):
        return None

    answer = obj.get("answer", obj.get("Answer", obj.get("label", obj.get("Label"))))
    explanation = obj.get("explanation", obj.get("Explanation", obj.get("rationale", obj.get("Rationale"))))

    if answer is None and explanation is None:
        return None

    ans = (answer or "")
    if isinstance(ans, str):
        ans = ans.strip().lower()
    else:
        ans = str(ans).strip().lower()

    return {
        "answer": ans,
        "explanation": _coerce_explanation(explanation),
    }

def _safe_parse_dict(s):
    """
    Parse a string into a dictionary using json.loads or ast.literal_eval.

    param: s (str): String representation of a dictionary.
    returns: obj (dict or None): Parsed dictionary if successful, else None.
    """
    if not s:
        return None

    ss = _strip_trailing_commas(s.strip())

    try:
        obj = json.loads(ss)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        obj = ast.literal_eval(ss)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return None

def extract_qids_from_text(text):
    """
    Function to keep a list of the seen qids to check if we are considering all of them
    param: text: string of text that is the output of the LLM
    returns: list of qids that have been seen
    """
    seen = set()
    out = []
    for m in _qid_key_pat.finditer(text or ""):
        qid = _normalize_qid(m.group("dq") or m.group("sq") or m.group("uq"))
        if qid and qid not in seen:
            seen.add(qid)
            out.append(qid)
    return out

def _extract_balanced_braces(text, start_idx):
    """
    Extract a balanced JSON-like dictionary starting at a given index.

    param: text (str): Raw LLM output text.
    param: start_idx (int): Index in text where '{' is expected.
    returns: substring (str or None): Extracted balanced {...} block.
    returns: end_idx_exclusive (int or None): End index (exclusive) of the block.
    """
    n = len(text)
    if start_idx < 0 or start_idx >= n or text[start_idx] != "{":
        return None, None

    i = start_idx
    depth = 0

    in_dq = False
    in_sq = False
    escape = False

    while i < n:
        c = text[i]

        if in_dq:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_dq = False
        elif in_sq:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == "'":
                in_sq = False
        else:
            if c == '"':
                in_dq = True
            elif c == "'":
                in_sq = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx:i+1], i+1

        i += 1

    return None, None

def _recover_qids(text, wanted_qids=None):
    """
    Recover qID-keyed JSON objects from raw text.

    param: text (str): Raw LLM response text.
    param: wanted_qids (set or None): Optional set of qIDs to restrict recovery.
    returns: recovered (dict): Mapping from recovered qIDs to parsed payloads.
    returns: found_but_unparseable (set): qIDs detected but not successfully parsed.
    """
    wanted = set(_normalize_qid(q) for q in wanted_qids) if wanted_qids else None

    recovered = {}
    bad = set()

    i = 0
    n = len(text)

    while True:
        m = _qid_key_pat.search(text, i)
        if not m:
            break

        qid = _normalize_qid(m.group("dq") or m.group("sq") or m.group("uq"))
        if wanted is not None and qid not in wanted:
            i = m.end()
            continue

        j = m.end()
        while j < n and text[j].isspace():
            j += 1

        # payload must start with "{"
        if j >= n or text[j] != "{":
            i = m.end()
            continue

        obj_str, endj = _extract_balanced_braces(text, j)
        if not obj_str:
            bad.add(qid)
            i = m.end()
            continue

        parsed = _safe_parse_dict(obj_str)
        if parsed is None:
            bad.add(qid)
            i = endj
            continue

        norm = _normalize_payload(parsed)
        if norm is None:
            bad.add(qid)
        else:
            recovered[qid] = norm

        i = endj

    return recovered, bad

def _ingest_top_level_json_if_present(text):
    """
    Extract top-level qID-keyed JSON objects from raw LLM text.

    param: text (str): Raw LLM response (may contain code-fenced JSON).
    returns: merged (dict): Mapping from normalized qIDs to normalized prediction payloads.
    """
    # Try to locate code-fenced json first
    fence_pat = re.compile(r"```+\s*(?:json)?\s*(.*?)\s*```+", flags=re.IGNORECASE | re.DOTALL)
    blocks = fence_pat.findall(text or "")

    candidates = blocks[:]
    # also try the entire text as a last resort (some outputs are pure JSON)
    candidates.append(text or "")

    merged = {}

    for raw in candidates:
        raw = (raw or "").strip()
        if not raw:
            continue

        obj = None
        try:
            obj = json.loads(_strip_trailing_commas(raw))
        except Exception:
            obj = None

        if not isinstance(obj, dict):
            continue

        keys = [str(k).lower() for k in obj.keys()]
        qid_keys = [k for k in keys if _is_qid_key(k)]

        # must look like a real qid map, not a random inner dict
        if len(qid_keys) < 5:
            continue

        for k, v in obj.items():
            qid = _normalize_qid(str(k))
            if not _is_qid_key(qid):
                continue
            norm = _normalize_payload(v)
            if norm is not None:
                merged[qid] = norm

    return merged

def parse_llm_txt_to_qid_json(text, expected_total=None):
    """
    Parse raw LLM text output into a qID-keyed JSON dictionary.

    param: text (str): Raw LLM response text.
    param: expected_total (int or None): Optional expected number of qID entries.
    returns: parsed (dict): Mapping from qIDs to normalized prediction payloads.
    """
    merged = {}

    # 1. safe ingest of actual qid->payload dicts
    top = _ingest_top_level_json_if_present(text)
    merged.update(top)

    # Expected qids list if provided
    wanted = None
    if expected_total is not None:
        wanted = [f"q{i:04d}" for i in range(1, int(expected_total) + 1)]

    # 2. recover missing via scanning
    if wanted is None:
        recovered, bad = _recover_qids(text)
        for qid, payload in recovered.items():
            if qid not in merged:
                merged[qid] = payload
    else:
        missing_now = [q for q in wanted if q not in merged]
        recovered, bad = _recover_qids(text, wanted_qids=missing_now)
        for qid, payload in recovered.items():
            merged[qid] = payload

    # Build diagnostics
    all_found_qids = set(extract_qids_from_text(text))
    parsed_qids = set(merged.keys())

    missing = []
    if wanted is not None:
        for q in wanted:
            if q not in parsed_qids:
                missing.append(q)

    found_not_parsed = sorted(list(all_found_qids - parsed_qids))

    stats = {
        "parsed": len(parsed_qids),
        "expected": expected_total,
        "missing": missing,
        "found_not_parsed": found_not_parsed,
        "found_not_parsed_count": len(found_not_parsed),
    }

    return merged, stats

def generate(fixed_prompt, examples, amount, client, generated_file, model,
             delay=0.8, encoding="utf-8", save_every=25):
    """
    Generate LLM responses for batched prompts and save results to file.

    param: fixed_prompt (str): Instruction prefix added to each batch.
    param: examples (list): List of formatted problem strings.
    param: amount (int): Total number of problems to process.
    param: client (object): LLM API client used for generation.
    param: generated_file (str): Output filename for saving responses.
    param: model (str): Name of the model to use.
    param: delay (float): Delay (seconds) between API calls.
    param: encoding (str): File encoding for output.
    param: save_every (int): Save intermediate results every N generations.
    returns: results (dict): Dictionary containing all generated outputs.
    """
    with open(fixed_prompt + ".txt", "r", encoding=encoding) as file:
        fixed = file.read()

    models = model if isinstance(model, list) else [model]
    paths_by_model = {}

    for model_name in models:
        out_txt = f"{generated_file}_{model_name}.txt"
        out_json = f"final_{generated_file}_{model_name}.json"
        out_partial = f"{generated_file}_{model_name}_partial.json"

        # reset txt
        open(out_txt, "w", encoding=encoding).close()

        merged = {}

        prompts = pc.construct_prompt(fixed, examples, amount)

        for i, prompt in enumerate(prompts, start=1):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                chunk = (response.text or "") + "\n"
            except Exception as e:
                print("Error at", i, e)
                time.sleep(10)
                continue

            # write raw text always
            with open(out_txt, "a", encoding=encoding) as resp:
                resp.write(chunk)

            chunk_parsed, _ = parse_llm_txt_to_qid_json(chunk, expected_total=None)

            for qid, payload in chunk_parsed.items():
                if _is_qid_key(qid):
                    merged[qid] = payload

            # periodic checkpoint
            if (i % save_every) == 0:
                with open(out_partial, "w", encoding="utf-8") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)
                print(f"[{model_name}] checkpoint {i}/{len(prompts)} -> {len(merged)} qids")

            time.sleep(delay)

        # final pass: parse whole file to recover anything missed 
        with open(out_txt, "r", encoding=encoding) as f:
            full_text = f.read()

        final_merged, stats = parse_llm_txt_to_qid_json(full_text, expected_total=amount)

        for qid, payload in merged.items():
            if qid not in final_merged:
                final_merged[qid] = payload

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(final_merged, f, ensure_ascii=False, indent=2)

        # diagnostics
        print(f"[{model_name}] wrote {out_json} with {len(final_merged)}/{amount} qids")
        if stats["missing"]:
            print(f"[{model_name}] missing {len(stats['missing'])}. First 30: {stats['missing'][:30]}")
        if stats["found_not_parsed_count"]:
            print(f"[{model_name}] found-but-unparsed {stats['found_not_parsed_count']}. First 30: {stats['found_not_parsed'][:30]}")

        paths_by_model[model_name] = out_json

    if isinstance(model, list):
        return paths_by_model
    return paths_by_model[models[0]]


if __name__ == "__main__":

    txt_path = "LLM_auto_responses_gemini-2.5-flash.txt"
    expected_total = 985

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    merged, stats = parse_llm_txt_to_qid_json(text, expected_total=expected_total)

    base = os.path.splitext(txt_path)[0]
    out_json = base + ".json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Parsed {stats['parsed']}/{expected_total} qids from {txt_path}")
    if stats["missing"]:
        print(f"Missing {len(stats['missing'])} qids. First 30: {stats['missing'][:30]}")
    if stats["found_not_parsed_count"]:
        print(f"Found-but-unparsed {stats['found_not_parsed_count']} qids. First 30: {stats['found_not_parsed'][:30]}")
    print(f"Wrote {out_json}")