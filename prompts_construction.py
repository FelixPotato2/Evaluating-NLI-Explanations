import evaluate_LLMs as ev
import math

def construct_prompt(fixed, examples, amount):
    if not isinstance(fixed, str):
        raise ValueError('First argument must be a string')
    for i, ex in enumerate(examples):
        if not isinstance(ex, str):
            raise ValueError('Elements of second argument must be a string')
        if amount <= 50:
            return [fixed + "\n" + ex for ex in examples]
        p = []
        step = max(1, math.ceil(amount / 10)) 
        if i % step == 0:
            p.append(fixed + '\n' + ex)
        elif i % step != 0:
            p[-1] = p[-1] + '\n' + ex
    return p






