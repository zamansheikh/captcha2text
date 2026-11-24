"""Work out the sums on math captchas ("4+3-5" -> 2).

Only whole numbers and + - × (also written x or *) are accepted, multiplication first, then
left to right. Anything else, such as two operators in a row or a missing number, is not a
sum and gives None. Python's eval is never used.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\d+|[+\-×xX*]")
_MULTIPLY = set("×xX*")


def solve_expression(text: str) -> int | None:
    s = "".join(text.split())
    if not s or len(s) > 40:
        return None
    tokens = _TOKEN.findall(s)
    if "".join(tokens) != s or len(tokens) % 2 == 0:
        return None
    numbers, operators = tokens[0::2], tokens[1::2]
    if not all(n.isdigit() and len(n) <= 6 for n in numbers) or any(o.isdigit() for o in operators):
        return None
    terms = [int(numbers[0])]
    for op, n in zip(operators, numbers[1:]):
        if op in _MULTIPLY:
            terms[-1] *= int(n)
        else:
            terms.append(int(n) if op == "+" else -int(n))
    return sum(terms)
