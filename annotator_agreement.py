# # # f = open("manual_annotations.txt")
# # # text = f.read()
# # # blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
# # # print(blocks[0:3])

# # import ast
# # import re

# # def load_dictionaries_from_txt(path):
# #     with open(path, "r", encoding="utf-8", errors="replace") as f:
# #         raw_text = f.read()

# #     # with open(path, "r") as f:
# #     #     raw_text = f.read()

# #     # 1. Normalize smart quotes → normal quotes
# #     replacements = {
# #         "“": '"',
# #         "”": '"',
# #         "’": "'",
# #         "‘": "'"
# #     }
# #     for k, v in replacements.items():
# #         raw_text = raw_text.replace(k, v)

# #     # 2. Split into dictionary blocks
# #     blocks = re.findall(r"\{.*?\}", raw_text, flags=re.DOTALL)

# #     parsed_dicts = []

# #     for block in blocks:
# #         try:
# #             # 3. Safely parse as Python literal
# #             parsed = ast.literal_eval(block)
# #             parsed_dicts.append(parsed)
# #         except Exception as e:
# #             print("Failed to parse block:")
# #             print(block)
# #             print("Error:", e)

# #     return parsed_dicts


# # data = load_dictionaries_from_txt("manual_annotations.txt")

# # print(len(data))          # number of parsed dictionaries
# # print(data[0])            # first dictionary


# import ast
# import re

# def load_dictionaries_from_txt(path):
#     with open(path, "r", encoding="cp1252") as f:
#         raw_text = f.read()

#     # 1. Remove non-breaking spaces completely
#     raw_text = raw_text.replace("\xa0", " ")

#     # 2. Normalize smart quotes
#     raw_text = raw_text.replace("“", '"').replace("”", '"')
#     raw_text = raw_text.replace("‘", "'").replace("’", "'")

#     # 3. Find dictionary blocks
#     blocks = re.findall(r"\{.*?\}", raw_text, flags=re.DOTALL)

#     parsed_dicts = []

#     for block in blocks:
#         try:
#             # 4. Quote image keys (e.g. 123.jpg#0r1e → "123.jpg#0r1e")
#             block = re.sub(
#                 r"\{\s*([^\s:]+)\s*:",
#                 r'{"\1":',
#                 block
#             )

#             parsed = ast.literal_eval(block)
#             parsed_dicts.append(parsed)

#         except Exception as e:
#             # Skip truly broken entries, but report them
#             print("Skipping broken block:")
#             print(block)
#             print("Reason:", e)
#             print("-" * 40)

#     return parsed_dicts


# data = load_dictionaries_from_txt("manual_annotations.txt")

# print(len(data))     # should now be > 0
# print(data[0])

import json
from evaluate_LLMs import Get_manual_evaluation_problems

# Read from file and parse JSON
with open("gold.json", "r") as f:
    gold_answers = json.load(f)

with open("manual.json", "r") as f2:
    script_answers =json.load(f2)

def get_len_lists(file):
    left_len_list =[]
    right_len_list =[]
    for dict in gold_answers:
        for answers in dict.values(): #this is only one...
            #print("answers")
            #print(answers)
            for answer_group in answers:
                #print("group")
                #print(answer_group)
                for answer in answer_group["left"]:
                    #print(answer.split())
                    left_len_list.append(len(answer.split()))
                for answer in answer_group["right"]:
                    right_len_list.append(len(answer.split()))
                #left_len_list.append(len(answer_group["left"]))
            # right_len_list.append(len(answer_group["right"]))
    return left_len_list, right_len_list

def get_len_lists_script():
    sript_pr, script_answers = Get_manual_evaluation_problems()
    left_len_list =[]
    right_len_list =[]
    for answers in script_answers.values(): #this is only one...
            #print("answers")
            #print(answers)
        for answer_group in answers:
                #print("group")
                #print(answer_group)
            for answer in answer_group["left"]:
                    #print(answer.split())
                left_len_list.append(len(answer))
            for answer in answer_group["right"]:
                right_len_list.append(len(answer))
                #left_len_list.append(len(answer_group["left"]))
            # right_len_list.append(len(answer_group["right"]))
    return left_len_list, right_len_list

    

def print_len_info(left_len_list, right_len_list):
    avg_len_left = sum(left_len_list) / len(left_len_list)
    avg_len_right = sum(right_len_list)/len(right_len_list)
    print(f"max left: {max(left_len_list)}")
    print(f"max right: {max(right_len_list)}")
    print(f"average len left: {avg_len_left}")
    print(f"average len right: {avg_len_right}")

left_len_list_g, right_len_lists_g = get_len_lists(gold_answers)
left_len_list_s, right_len_lists_s = get_len_lists_script()
print("gold set----------------")
print_len_info(left_len_list_g, right_len_lists_g)
print("script set -----------------")
print_len_info(left_len_list_s, right_len_lists_s)




#print(data)
#print(type(data))


#print(len(data))




# for ID in script_answers:
#     print(ID)
#     print(script_answers[ID])
# for problem in data:
#     for ID in problem.keys(): #its a bit dumb in how we structured it but this should be just one
#         script_ans = script_answers[ID]
#         for answer_dict in problem[ID]:
#             left_set = set()
#             right_set =set()
#             for word in answer_dict["left"]:
#                 if word not in ["a", "an", "the"]:
#                     left_set.add(word.lower())
#             for word in answer_dict["right"]:
#                 if word not in ["a", "an", "the"]:
#                     right_set.add(word.lower())
            