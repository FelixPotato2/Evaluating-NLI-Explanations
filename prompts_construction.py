import math

def construct_prompt(fixed, examples, amount):
    if not isinstance(fixed, str):
        raise ValueError("fixed must be a string")
    if not examples:
        raise ValueError("examples is empty")

    examples = examples[:amount]
    print(f'Number of examples: {len(examples)}\n')

    if amount <= 50:
        return [fixed + "\n" + ex for ex in examples]

    if amount >= 1000:
        step = 100
    else:
        print('Number of examples is between 50 and 1000, so we proceed with 10%\n')
        step = max(1, math.ceil(len(examples) / 10))

    p = []
    for i, ex in enumerate(examples):
        if i % step == 0 or not p:
            p.append(fixed + "\n" + ex)
        else:
            p[-1] += "\n" + ex
    return p

