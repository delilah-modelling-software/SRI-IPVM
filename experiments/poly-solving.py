import sys
import numpy as np
import scipy as sp
import scipy.stats as ss


x1 = -2
x2 = -1
x3 =  1
x4 =  2

examples_linear    = []
examples_quadratic = [(x1, x2), (x1, x1)]
examples_cubic     = [(x1, x2, x3), (x1, x1, x2), (x1, x1, x1)]
examples_quartic   = [(x1, x2, x3, x4), (x1, x1, x2, x3), (x1, x1, x2, x2), (x1, x1, x1, x2), (x1, x1, x1, x1)]

def construct_quadratic(roots):
    assert(len(roots) == 2)
    coeffs = [roots[0] * roots[1], -(roots[0] + roots[1]), 1]
    return coeffs

def construct_cubic(roots):
    assert(len(roots) == 3)
    coeffs = [-(roots[0] * roots[1] * roots[2]), roots[0] * roots[1] + roots[1] * roots[2] + roots[2] * roots[0], -(roots[0] + roots[1] + roots[2]), 1]
    return coeffs

def construct_quartic(roots):
    assert(len(roots) == 4)
    assert(False)
    return None

def eval_linear(coeffs, x):
    return coeffs[1] * x + coeffs[0]

def eval_quadratic(coeffs, x):
    return (coeffs[2] * x + coeffs[1]) * x + coeffs[0]

def eval_cubic(coeffs, x):
    p = coeffs[3] * x + coeffs[2]
    p = p * x + coeffs[1]
    p = p * x + coeffs[0]
    return p
    # return ((coeffs[3] * x + coeffs[2]) * x + coeffs[1])*x + coeffs[0]

def eval_quartic(coeffs, x):
    return (((coeffs[4] * x + coeffs[3]) * x + coeffs[2]) * x + coeffs[1])*x + coeffs[0]

def eval_poly(coeffs, x):
    n = len(coeffs)
    assert(n >= 1)
    p = coeffs[-1] * x + coeffs[-2]
    for n in range(3, n + 1):
        p = p * x + coeffs[-n]
    return p

def solve_quadratic(coeffs):
    a = coeffs[2]
    b = coeffs[1]
    c = coeffs[0]
    disc = b**2 - 4 * a * c
    if disc < 0:
        return None

    term1 = -b / (2 * a)
    term2 = np.sqrt(disc) / (2 * a)
    return term1 - term2, term1 + term2

def simplify_cubic(coeffs):
    #
    # Monicise
    #
    n       = len(coeffs)
    divisor = coeffs[-1]
    if divisor != 1:
        for k in range(n - 1):
            coeffs[k] /= divisor
        coeffs[-1] = 1

    print("Monic cubic {}".format(coeffs))

    #
    # Depress
    #
    a0 = coeffs[0]
    a1 = coeffs[1]
    a2 = coeffs[2]

    h         = -a2 / 3
    coeffs[0] = eval_cubic([a0, a1, a2, 1], h)
    coeffs[1] = eval_quadratic([a1, 2 * a2, 3], h)
    coeffs[2] = eval_linear([a2, 3], h)

    print("Depressed cubic: subtracting h {} from the independent variable yields {}".format(h, coeffs))

    return coeffs, h

def simplify_quartic(coeffs):
    #
    # Monicise
    #
    n       = len(coeffs)
    divisor = coeffs[-1]
    if divisor != 1:
        for k in range(n - 1):
            coeffs[k] /= divisor
        coeffs[-1] = 1

    print("Monic quartic {}", coeffs)

    #
    # Depress
    #
    h = -coeffs[3] / 4
        
    return None

def solve_cubic(coeffs):
    coeffs, shift = simplify_cubic(coeffs)
    h = shift

    #
    # Now solve the depressed monic cubic.
    #
    p = coeffs[1]
    q = coeffs[0]
    disc = -(4 * coeffs[1]**3 - 27 * coeffs[0]**2)
    print("Cubic eqn with coefficients {} has discriminant {}".format(coeffs, disc))

    if disc > 0:
        p = coeffs[1]
        q = coeffs[0]
        term1 = -q / 2
        term2 = np.sqrt(disc)

        u1 = term1 - term2
        u2 = term1 + term2
        print("Cubic eqn with coefficients {} has soln {}^1/3 + {}^1/3".format(coeffs, u1, u2))

        if u1 < 0:
            x1 = -((-u1)**(1./3))
        else:
            x1 = u1**(1./3)
        if u2 < 0:
            x2 = -((-u2)**(1./3))
        else:
            x2 = u2**(1./3)
        print("Cubic eqn with coefficients {} has partial solns {}, {}".format(coeffs, x1, x2))

        x = x1 + x2
        print("Cubic eqn with coefficients {} has soln {} + h {} = {}".format(coeffs, x, h, x + h))

    return None

def solve_quartic(coeffs):
    coeffs, shift = simplify_quartic(coeffs)

    #
    # Now solve the depressed monic quartic.
    #
    
    return None

if  __name__ == "__main__":
    degree_descriptions = ["constant", "linear", "quadratic", "cubic", "quartic"]
    examples = [None, None, examples_quadratic, examples_cubic, examples_quartic]
    constructors = [None, None, construct_quadratic, construct_cubic, construct_quartic]
    solvers  = [None, None, solve_quadratic, solve_cubic, solve_quartic]

    for i in range(2, 4): # range(2, 5):
        print("#")
        print("# Begin {}".format(degree_descriptions[i]))
        print("#")
        print("")
        
        active_examples    = examples[i]
        active_constructor = constructors[i]
        active_solver      = solvers[i]
        
        for j in range(len(active_examples)):
            example = active_examples[j]
            print("Given roots are: {}".format(example))

            coefficients = active_constructor(example)
            print("Coefficients are: {}".format(coefficients))

            for root in example:
                value = eval_poly(coefficients, root)
                print("Value of polynomial at root {} is {}".format(root, value))
            
            str = "Polynomial = {}".format(coefficients[0])
            if len(coefficients) > 1:
                str += " + {}x".format(coefficients[1])
            for k in range(2, len(coefficients)):
                str += " + {}x^{}".format(coefficients[k], k)
            print(str)
            
            soln = active_solver(coefficients)
            print("Found roots are: {}".format(soln))
            print("")
            
        print("")

