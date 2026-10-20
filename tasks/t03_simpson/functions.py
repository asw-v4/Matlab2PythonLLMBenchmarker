"""The functions the tests integrate, by name. The Octave versions are in generate_expected.m;
a test checks the two agree. The harness looks these up from the `f` input of each case."""
import numpy as np

FUNCS = {
    "cubic": lambda x: x ** 3,
    "gauss": lambda x: np.exp(-x ** 2),
    "sine": lambda x: np.sin(x),
    "quartic": lambda x: x ** 4,
    "lorentz": lambda x: 1.0 / (1.0 + x ** 2),
}

# exact integrals, used only to sanity-check the answer key
TRUE_VALUES = {
    ("cubic", 0.0, 2.0): 4.0,
    ("sine", 0.0, float(np.pi)): 2.0,
    ("quartic", -1.0, 3.0): (3.0 ** 5 - (-1.0) ** 5) / 5.0,
    ("lorentz", 0.0, 1.0): float(np.pi) / 4.0,
}
