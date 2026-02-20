import json
from evaluation import Get_manual_evaluation_problems
from evaluation import get_LLM_problems
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open("gold.json", "r") as f:
    gold_answers = json.load(f)

def get_len_lists(file):
    """
    Compute token lengths of left and right gold relations.

    param: file (any): Placeholder argument (not used).
    returns: left_len_list (list): Token lengths of left-side relations.
    returns: right_len_list (list): Token lengths of right-side relations.
    """
    left_len_list =[]
    right_len_list =[]
    for dict in gold_answers:
        for answers in dict.values(): #this is only one
            for answer_group in answers:
                for answer in answer_group["left"]:
                    left_len_list.append(len(answer.split()))
                for answer in answer_group["right"]:
                    right_len_list.append(len(answer.split()))
                
    return left_len_list, right_len_list

def get_len_lists_script(script_answers):
    """
    Compute token lengths of left and right relations from script answers.

    param: script_answers (dict): Dictionary mapping pairIDs to answer groups.
    returns: left_len_list (list): Lengths of left-side token lists.
    returns: right_len_list (list): Lengths of right-side token lists.
    """
    left_len_list =[]
    right_len_list =[]
    for answers in script_answers.values(): #this is only one
        for answer_group in answers:
            for answer in answer_group["left"]:
                left_len_list.append(len(answer))
            for answer in answer_group["right"]:
                right_len_list.append(len(answer))
    return left_len_list, right_len_list

    
def print_len_info(left_len_list, right_len_list):
    """
    Print max and average lengths for left and right relations.

    param: left_len_list (list): Lengths of left-side relations.
    param: right_len_list (list): Lengths of right-side relations.
    returns: None
    """
    avg_len_left = sum(left_len_list) / len(left_len_list)
    avg_len_right = sum(right_len_list)/len(right_len_list)
    print(f"max left: {max(left_len_list)}")
    print(f"max right: {max(right_len_list)}")
    print(f"average len left: {avg_len_left}")
    print(f"average len right: {avg_len_right}")


def make_len_plot(d1, d2, d3, d4, d5, d6):
    """
    Create length comparison plot for multiple distributions.

    param: d1 (list): First length distribution.
    param: d2 (list): Second length distribution.
    param: d3 (list): Third length distribution.
    param: d4 (list): Fourth length distribution.
    param: d5 (list): Fifth length distribution.
    param: d6 (list): Sixth length distribution.
    returns: None
    """

    d = [d1, d2, d3, d4, d5, d6]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_ylabel("Word count")
    colors = ['#80b3ff', '#0052cc', 
          '#ccb3ff', '#7733ff', "#ffe066", "#cca300"]
    bp = ax.boxplot(d, labels = ["Complete left", "Complete  right", "Auto templates left", "Auto templates right", "Gold templates left", "Gold templates right"], patch_artist=True)

    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
    for median in bp['medians']:
        median.set_linewidth(2)
        median.set_color("black")

    plt.show()

   
def make_perfection_plot():
    """
    Create a stacked bar plot showing perfect vs. non-perfect template matches.

    param: None
    returns: None
    """

    N = 2
    useable = (31, 21)
    unuseable = (0, 8)
    ind = np.arange(N) * 0.17
    width = 0.1

    fig, ax = plt.subplots(figsize =(10, 7))
    p1 = plt.bar(ind, useable, width, color = '#80b3ff' )
    p2 = plt.bar(ind, unuseable, width, bottom = useable, color = '#0052cc')

    plt.ylabel('Number of answer templates', fontsize = 20)
    plt.xticks(ind, ('Perfect match', 'Non perfect match'), fontsize = 20)
    plt.legend((p1[0], p2[0]), ('Minor differences', 'Problematic differences'), fontsize = 18)
    ax.set_ylim(0, 35)
    ax.tick_params(axis='y', labelsize=20)

    plt.tight_layout()
    plt.show()


def make_boring_plot():
    """
    Create and display the complexity and phrase-type distribution plot.

    param: None
    returns: None
    """

    N = 2

    noun = np.array((36, 10))
    verb = np.array((3, 3))
    mix  = np.array((1, 2))
    amb  = np.array((2, 2))
    sen  = np.array((2, 0))

    ind = np.arange(N) * 0.9   
    width = 0.15               

    fig, ax = plt.subplots(figsize=(10, 7))

    p1 = ax.bar(ind - 2*width, noun, width, color='#80b3ff', label='NP')
    p2 = ax.bar(ind - width,  verb, width, color='#ccb3ff', label='VP')
    p3 = ax.bar(ind,          mix,  width, color='#ff99bb', label='Mix')
    p4 = ax.bar(ind + width,  amb,  width, color='#ffcc99', label='Ambiguous')
    p5 = ax.bar(ind + 2*width, sen, width, color='#ffe066', label='Sentence')

    for bars in [p1, p2, p3, p4, p5]:
        ax.bar_label(bars, fontsize=16, padding=3)

    ax.set_ylabel('Number of answer templates', fontsize=20)
    ax.set_xticks(ind)
    ax.set_xticklabels(('Simple', 'Complex'), fontsize=20)
    ax.tick_params(axis='y', labelsize=20)

    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

    
    
merged_df = pd.read_csv("merged_entailment.csv")
_, answers_all, _, _= get_LLM_problems(merged_df, 771)
left_len_list_all_s, right_len_list_all_s = get_len_lists_script(answers_all)
_, script_answers = Get_manual_evaluation_problems(print_results = False, print_answers= False) #to print the information of the 60 annotated problems and answer templates, make these both true
left_len_list_s, right_len_lists_s = get_len_lists_script(script_answers)
left_len_list_g, right_len_lists_g = get_len_lists(gold_answers)

print("gold set----------------")
print_len_info(left_len_list_g, right_len_lists_g)
print("script set -----------------")
print_len_info(left_len_list_s, right_len_lists_s)
print("all script -------")
print_len_info(left_len_list_all_s, right_len_list_all_s)


#uncomment these to create various plots
#make_len_plot(left_len_list_all_s, right_len_list_all_s, left_len_list_s, right_len_lists_s, left_len_list_g, right_len_lists_g)
make_perfection_plot()
make_boring_plot()







