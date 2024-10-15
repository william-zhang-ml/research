"""Financial modeling code. """


def get_eq_beta(divyield: float, base: float = 0.07):
    excess = base - divyield
    return excess / base
