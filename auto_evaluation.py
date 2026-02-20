import generation as g
import evaluation as ev
import pandas as pd
from google import genai

def main():
    """
        Main function to run final evaluation.    
    """

    n = 985
    alias_to_full, gold_answers, problems = ev.Get_prompts_for_LLM(n)
    gen_client = genai.Client()
    fixed_prompt = "fixed"
    models = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]
    gen_file_base = "LLM_auto_responses"

    paths_by_model = g.generate(
        fixed_prompt=fixed_prompt,
        examples=problems,
        amount=n,
        client=gen_client,
        generated_file=gen_file_base,
        model=models,
        delay=10,
    )

    if isinstance(paths_by_model, dict):
        for k, v in list(paths_by_model.items()):
            if isinstance(v, dict) and "json" in v:
                paths_by_model[k] = v["json"]

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
        print("Percentage entailment:", out["perc_entailment"])
        print("Pairs improved ONLY by allowing 'implies':", out["implies_only_wrong_pairs"])

        print("\n--- TYPE OF ONLY (current evaluator) ---")
        print("Strict scores:", out["type_of_only"]["strict"])
        print("Loose scores:", out["type_of_only"]["loose"])
        print("Length metrics:", out["type_of_only"]["len_metrics"])
        print("Length Scores:", out["type_of_only"]["len_scores"])
        print("LLM Average Length:", out["type_of_only"]["avg_len"])
        print("Annotator's Average Length", out["type_of_only"]["gold_avg_len"])

        print("\n--- TYPE OF + IMPLIES (relaxed) ---")
        print("Strict scores:", out["type_of_plus_implies"]["strict"])
        print("Loose scores:", out["type_of_plus_implies"]["loose"])
        print("Length metrics:", out["type_of_plus_implies"]["len_metrics"])
        print("Length Scores:", out["type_of_plus_implies"]["len_scores"])
        print("Average Length:", out["type_of_plus_implies"]["avg_len"])
        print("Annotator's Average Length", out["type_of_plus_implies"]["gold_avg_len"])

def evaluate_existing_runs(json_paths, n=770):
    """
    Evaluate previously generated LLM outputs without regenerating.

    param: json_paths (dict): {model_name: path_to_json}
    param: n (int): Number of examples used when prompts were generated (must match the original run so id_map aligns)
    """

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
        print("\nPercentage entailment:", out["perc_entailment"])
        print("Pairs improved ONLY by allowing 'implies':",
              out["implies_only_wrong_pairs"])

        print("\n--- TYPE OF ONLY (current evaluator) ---")
        print("Strict scores:", out["type_of_only"]["strict"])
        print("Loose scores:", out["type_of_only"]["loose"])
        print("Length metrics:", out["type_of_only"]["len_metrics"])
        print("Length Scores:", out["type_of_only"]["len_scores"])
        print("LLM Average Length:", out["type_of_only"]["avg_len"])
        print("Annotator's Average Length", out["type_of_only"]["gold_avg_len"])

        print("\n--- TYPE OF + IMPLIES (relaxed) ---")
        print("Strict scores:", out["type_of_plus_implies"]["strict"])
        print("Loose scores:", out["type_of_plus_implies"]["loose"])
        print("Length metrics:", out["type_of_plus_implies"]["len_metrics"])
        print("Length Scores:", out["type_of_plus_implies"]["len_scores"])
        print("Average Length:", out["type_of_plus_implies"]["avg_len"])
        print("Annotator's Average Length", out["type_of_plus_implies"]["gold_avg_len"])


if __name__ == "__main__":
    flag = True #If true run the existing json files for evaluation, if False prompt gemini
    if flag:
        existing_runs = {
            "gemini-2.5-flash": "final_LLM_auto_responses_gemini-2.5-flash.json",
            "gemini-2.5-pro": "final_LLM_auto_responses_gemini-2.5-pro.json"
        }
        evaluate_existing_runs(existing_runs, n=985)
    else:
        main()
