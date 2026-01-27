import math

def construct_prompt(fixed, examples, amount):
    if not isinstance(fixed, str):
        raise ValueError("First argument must be a string")
    if not examples:
        raise ValueError("examples is empty")

    # If you want <=50 to be "one prompt per example"
    if amount <= 50:
        return [fixed + "\n" + ex for ex in examples]

    # Otherwise group into ~10 prompts
    p = []
    step = max(1, math.ceil(len(examples) / 10))

    for i, ex in enumerate(examples):
        if not isinstance(ex, str):
            raise ValueError("Elements of second argument must be strings")

        if i % step == 0 or not p:
            p.append(fixed + "\n" + ex)
        else:
            p[-1] += "\n" + ex

    return p