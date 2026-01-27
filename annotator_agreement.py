import json
from evaluate_LLMs import Get_manual_evaluation_problems
from evaluate_LLMs import Get_prompts_for_LLM

# Read from file and parse JSON
with open("gold.json", "r") as f:
    gold_answers = json.load(f)

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

def get_len_lists_script(script_answers):
    
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


_, answers_all, _= Get_prompts_for_LLM(amount = 771)
left_len_list_all_s, right_len_list_all_s = get_len_lists_script(answers_all)
_, script_answers = Get_manual_evaluation_problems()
left_len_list_s, right_len_lists_s = get_len_lists_script(script_answers)
left_len_list_g, right_len_lists_g = get_len_lists(gold_answers)
print("gold set----------------")
print_len_info(left_len_list_g, right_len_lists_g)
print("script set -----------------")
print_len_info(left_len_list_s, right_len_lists_s)
print("all script -------")
print_len_info(left_len_list_all_s, right_len_list_all_s)




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
            