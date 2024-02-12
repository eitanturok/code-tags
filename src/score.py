import numpy as np


def estimator(n: int, c: int, k: int) -> float:
    """Computes the pass@k metric.
    Given the number of generated samples, n, the number of correct samples, c, and the k of interest,
    this function calculates pass@k as 1 - comb(n - c, k) / comb(n, k) as per the definition of
    pass@k in the HumanEval paper (https://arxiv.org/abs/2107.03374) and it's associated implementation:
    https://github.com/openai/human-eval.
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def scorer(all_pass_tests, k):
    n, c = 0, 0
    for pass_tests in all_pass_tests.values():
        c += int(pass_tests)
        n += 1

    score = estimator(n, c, k)
    return score
