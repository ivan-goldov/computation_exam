import numpy as np
from typing import Callable, Union
from functools import partial
import tabulate
import matplotlib.pyplot as plt

def sf(n: int):
    return [(2*i+1) / (2 * n) for i in range(n)]
 
def quad_formula(f: Callable, n: int = 10, start: float = 0., end: float = 1.) -> np.array:
    h = (end - start) / n
    return np.array([h * f(start + h * (i + 0.5)) for i in range(n)])

def build_matrix(n: int = 10, start: float = 0., end: float = 1., k: Callable = k) -> np.array:
    a = []
    h = (end - start) / n
    s = sf(n)
    for i in s:
        a.append([])
        for j in s:
            a[-1].append(h * k(i, j))
    return np.array(a)

def k(x: float, s: float) -> float:
    return np.exp(-2*x*s)

def k1(s: float, t: float) -> float:
    return (1 - np.exp(-2*(s+t))) / (2*(s + t))

def z1(x: float):
    return 1

def z2(x: float) -> np.array:
    return x * (1 - x)

def u1(x: float) -> float:
    return (1 - np.exp(-x))/x

def u2(x: float) -> float:
    if x == 0: return 1/6
    return (x + np.exp(-2*x) * x - 1 + np.exp(-2*x)) / (4 * x ** 3)

def solve_sle(a: np.array, b: np.array, n: int, alpha: float) -> np.array:
    a_hermite = a.T
    new_u = a_hermite @ b
    new_matrix = a_hermite @ a + np.eye(n) * alpha
    return np.linalg.solve(new_matrix, new_u)

def solve_first_method(n: int = 10, alpha: float = 0.00001, u: Callable = u1):
    a = build_matrix(n)
    s = sf(n)
    u_vector = [u(i) for i in s]
    z = solve_sle(a, u_vector, n, alpha)
    return z

def check_first_method():
    z_t = z2
    u = u2
    ans = []
    alpha_array = [10 ** (-13 + i) for i in range(13)]
    for n in range(1, 30):
        for alph in alpha_array:
            z = solve_first_method(n, u=u, alpha=alph)
            s = sf(n)
            z_arr = [z_t(i) for i in s]
            ans.append((n, np.linalg.norm(z-z_arr), alph))
    return ans


def print_first_method_results():
    table = check_first_method()
    # print('Первый метод')
    print(tabulate.tabulate(table, headers=['n', '||z-z_0||', 'alpha']))
    x1 = [i[0] for i in table]
    y1 = [i[1] for i in table]
    # plt.plot(x1, y1)

def select_best_alpha(array):
    best = []
    for n in range(1, 30):
        best_alpha, best_norm = 1e5, 1e5
        for it in array:
            if it[0] == n:
                if it[1] < best_norm:
                    best_norm = it[1]
                    best_alpha = it[2]
        best.append((n, best_norm, best_alpha))
    return best


def solve_second_method(n: int = 10, alpha1: float = 1e-10, alpha2: float = 1.5e-5, u: Callable = u2,
                        k1: Callable = k1, k: Callable = k, p=1, r=1):
    h = (1. - 0.) / n
    a = []
    s = [i/n for i in range(n+1)]
    for i in range(1, n):
        a.append([])
        for j in range(1, n):
            a[-1].append(k1(s[i], s[j]) / n)

    b = [[0 for _ in range(n-1)] for i in range(n-1)]
    h = (1. - 0.) / n
    for i in range(n-1):
        b[i][i] += ((p+p) / h ** 2) + r
        if i != 0:
            b[i][i-1] -= p / h ** 2
        if i != n - 2:
            b[i][i+1] -= p / h ** 2
    a = np.array(a)
    b = np.array(b)
    # print('A')
    # print(a)
    # print('B')
    # print(b)
    c = a + alpha2 * b
    us = []
    for s_i in s[1:-1]:
        us.append(0)
        for j in range(len(s)):
            if j == 0:
                us[-1] += ( u(s[0]) * k(s[0], s_i) * (s[1] - s[0]) ) / 2 
            elif j == n:
                us[-1] += ( u(s[n]) * k(s[n], s_i) * (s[n] - s[n-1]) ) / 2
            else:
                us[-1] += (s[j+1]-s[j-1]) * k(s[j], s_i) * u(s[j])
    us = np.array(us)
    z = solve_sle(c, us, n-1, alpha1)
    return z

def check_second_method():
    z_t = z2
    ans = []
    alpha_array = [10 ** (-13 + i) for i in range(13)]
    for n in range(3, 30):
        for alph1 in alpha_array:
            for alph2 in alpha_array:
                z = solve_second_method(n, alpha1=alph1, alpha2=alph2)
                s = [i/n for i in range(n+1)][1:n]
                z_arr = [z_t(i) for i in s]
                ans.append((n, np.linalg.norm(z-z_arr), alph1, alph2))
    return ans

def print_second_method_results():
    table = check_second_method()
    res = select_best_alpha1_alpha2(table)
    print(tabulate.tabulate(res, headers=['n', '||z-z_0||', 'alpha_reg', 'alpha_euler']))
    x1 = [i[0] for i in res]
    y1 = [i[1] for i in res]
    plt.plot(x1, y1)


def select_best_alpha1_alpha2(array):
    best = []
    for n in range(3, 30):
        best_alpha1, best_alpha2, best_norm = 1e5, 1e5, 1e5
        for it in array:
            if it[0] == n:
                if it[1] < best_norm:
                    best_norm = it[1]
                    best_alpha1 = it[2]
                    best_alpha2 = it[3]
        best.append((n, best_norm, best_alpha1, best_alpha2))
    return best

# print_first_method_results()

res1 = select_best_alpha(check_first_method())

# print(tabulate.tabulate(res1, headers=['n', '||z-z_0||', 'alpha'], tablefmt='simple'))

# print_second_method_results();

res1 = select_best_alpha(check_first_method())

y1 = [it[1] for it in res1][1:]
y2 = [it[1] for it in res]
x = [it[0] for it in res]
plt.xlabel('n')
plt.ylabel('||z-z_0||')
plt.plot(x, y1, label='Первый метод')
plt.plot(x, y2, label='Второй метод')
plt.legend()
plt.show()