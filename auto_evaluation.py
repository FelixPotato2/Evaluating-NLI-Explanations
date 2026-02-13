# import generation as g
# import evaluation as ev
# from google import genai

# def main():
#     n = 770  # max is 771

#     # Load prompts / gold answers
#     alias_to_full, gold_answers, problems = ev.Get_prompts_for_LLM(n)

#     # GenAI client
#     gen_client = genai.Client()

#     # Prompt file base name (expects "<fixed_prompt>.txt" to exist)
#     fixed_prompt = "fixed"

#     # Models to run
#     models = [
#         "gemini-2.5-flash",
#         "gemini-2.5-pro",
#     ]

#     gen_file_base = "LLM_auto_responses"

#     # generate() returns dict: {model: final_json_path}
#     paths_by_model = g.generate(
#         fixed_prompt=fixed_prompt,
#         examples=problems,
#         amount=n,
#         client=gen_client,
#         generated_file=gen_file_base,
#         model=models,
#         delay=10,
#     )

#     for model_name, json_path in paths_by_model.items():
#         llm_answers = ev.read_json(json_path)

#         # NEW: checK_LLM returns a dict with two metric sets + counters
#         out = ev.checK_LLM(
#             llm_answers,
#             answers=gold_answers,
#             id_map=alias_to_full,
#         )

#         print("\n===================================")
#         print(f"MODEL: {model_name}")
#         print("-----------------------------------")
#         print("Number of pairIDs:", g.count_pairIDs(json_path))
#         print("Percentage entailment:", out["perc_entailment"])
#         print("Pairs improved ONLY by allowing 'implies':", out["implies_only_wrong_pairs"])

#         print("\n--- TYPE OF ONLY (current evaluator) ---")
#         print("Strict scores:", out["type_of_only"]["strict"])
#         print("Loose scores:", out["type_of_only"]["loose"])
#         print("Length metrics:", out["type_of_only"]["len_metrics"])
#         print("Length Scores:", out["type_of_only"]["len_scores"])

#         print("\n--- TYPE OF + IMPLIES (relaxed) ---")
#         print("Strict scores:", out["type_of_plus_implies"]["strict"])
#         print("Loose scores:", out["type_of_plus_implies"]["loose"])
#         print("Length metrics:", out["type_of_plus_implies"]["len_metrics"])
#         print("Length Scores:", out["type_of_plus_implies"]["len_scores"])

# if __name__ == "__main__":
#     main()

import generation as g
import evaluation as ev
import pandas as pd
from google import genai

def main():

    n = 985
    # Load prompts / gold answers
    alias_to_full, gold_answers, problems = ev.Get_prompts_for_LLM(n)

    # GenAI client
    gen_client = genai.Client()

    # Prompt file base name (expects "<fixed_prompt>.txt" to exist)
    fixed_prompt = "fixed"

    # Models to run
    models = [
        #"gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    gen_file_base = "LLM_auto_responses"

    # NEW generate(): if you pass a list of models, it returns {model_name: final_json_path}
    paths_by_model = g.generate(
        fixed_prompt=fixed_prompt,
        examples=problems,
        amount=n,
        client=gen_client,
        generated_file=gen_file_base,
        model=models,          # list[str]
        delay=10,
    )

    # Evaluate each model
    for model_name, json_path in paths_by_model.items():
        llm_answers = ev.read_json(json_path)

        out = ev.checK_LLM(
            llm_answers,
            answers=gold_answers,
            id_map=alias_to_full,
        )

        print("\n===================================")
        print(f"MODEL: {model_name}")
        print("-----------------------------------")
        print("JSON path:", json_path)
        #print("Number of pairIDs:", g.count_pairIDs(json_path))
        print("Percentage entailment:", out["perc_entailment"])
        print("Pairs improved ONLY by allowing 'implies':", out["implies_only_wrong_pairs"])

        print("\n--- TYPE OF ONLY (current evaluator) ---")
        print("Strict scores:", out["type_of_only"]["strict"])
        print("Loose scores:", out["type_of_only"]["loose"])
        print("Length metrics:", out["type_of_only"]["len_metrics"])
        print("Length Scores:", out["type_of_only"]["len_scores"])

        print("\n--- TYPE OF + IMPLIES (relaxed) ---")
        print("Strict scores:", out["type_of_plus_implies"]["strict"])
        print("Loose scores:", out["type_of_plus_implies"]["loose"])
        print("Length metrics:", out["type_of_plus_implies"]["len_metrics"])
        print("Length Scores:", out["type_of_plus_implies"]["len_scores"])

def evaluate_existing_runs(json_paths, n=770):
    """
    Evaluate previously generated LLM outputs without regenerating.

    Parameters
    ----------
    json_paths : dict
        {model_name: path_to_json}
    n : int
        Number of examples used when prompts were generated
        (must match the original run so id_map aligns)
    """

    # Rebuild alias map + gold answers (NO generation happens here)
    alias_to_full, gold_answers, _ = ev.Get_prompts_for_LLM(n)

    for model_name, json_path in json_paths.items():

        llm_answers = ev.read_json(json_path)

        out = ev.checK_LLM(
            llm_answers,
            answers=gold_answers,
            id_map=alias_to_full,
        )

        print("\n===================================")
        print(f"MODEL: {model_name}")
        print("-----------------------------------")
        print("JSON path:", json_path)
        #print("Number of pairIDs:", g.count_pairIDs(json_path))
        print("Percentage entailment:", out["perc_entailment"])
        print("Pairs improved ONLY by allowing 'implies':",
              out["implies_only_wrong_pairs"])

        print("\n--- TYPE OF ONLY (current evaluator) ---")
        print("Strict scores:", out["type_of_only"]["strict"])
        print("Loose scores:", out["type_of_only"]["loose"])
        print("Length metrics:", out["type_of_only"]["len_metrics"])
        print("Length Scores:", out["type_of_only"]["len_scores"])

        print("\n--- TYPE OF + IMPLIES (relaxed) ---")
        print("Strict scores:", out["type_of_plus_implies"]["strict"])
        print("Loose scores:", out["type_of_plus_implies"]["loose"])
        print("Length metrics:", out["type_of_plus_implies"]["len_metrics"])
        print("Length Scores:", out["type_of_plus_implies"]["len_scores"])

# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    flag = True
    if flag:
        existing_runs = {
            "gemini-2.5-flash": "final_LLM_auto_responses_gemini-2.5-flash.json",
            "gemini-2.5-pro": "final_LLM_auto_responses_gemini-2.5-pro.json"
        }
        evaluate_existing_runs(existing_runs, n=985)
    else:
        main()
