import math

def construct_prompt(fixed, examples, amount):
    """
    Construct the prompt to feed to the API by concatenating the fixed prompt with the appropriate number of examples.
    The number of examples appended to the text varies based on the amoung of problems: 
        - Up to 80 problems we append all of them directly to the fixed prompt
        - From 80 until 1000 examples we always consider 10% of the total examples to be appended afeter the fixed prompt
        - After 1000 we always use 100 examples

    param: fixed (str): string of text containing fixed prompt
    param: examples (list): list containing the examples we want to feed to the LLM in the correct format
    param: amount (int): total number of problems we have 
    """
    if not isinstance(fixed, str):
        raise ValueError("fixed must be a string")
    if not examples:
        raise ValueError("examples is empty")

    examples = examples[:amount]
    print(f'Number of examples: {len(examples)}\n')

    if amount <= 80:
        return [fixed + "\n" + ex for ex in examples]

    if amount >= 1000:
        step = 100
    else:
        print('Number of examples is between 80 and 1000, so we proceed with 10%\n')
        step = max(1, math.ceil(len(examples) / 10))

    p = []
    for i, ex in enumerate(examples):
        if i % step == 0 or not p:
            p.append(fixed + "\n" + ex)
        else:
            p[-1] += "\n" + ex
    return p

