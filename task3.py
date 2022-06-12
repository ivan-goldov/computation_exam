import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, cbrt, sqrt
from functools import lru_cache
from math import factorial, gamma
from sympy.abc import x

def f_d(n: int) -> float:
    if n == 0: return cbrt(x)/(sqrt(x)-1) 
    else: return diff(f_d(n-1), x)

def f(p: float, n: int = 0) -> float:
    return f_d(n).subs(x, p)

def w_f(n: int, t: float) -> float:
    p = n / t
    return (-1)**n * p**(n+1) / factorial(n) * f(p, n)

def w_improve(n: int, k: int, t: float) -> float:
    k = min(k, n)
    d = []
    for i in range(n, n-k, -1):
        d.append(i/n)
    c = []
    for i in range(k):
        tmp = []
        for j in range(k):
            if i != j:
                tmp.append(d[i] / (d[i]-d[j]))
        c.append(np.prod(tmp))
    s = 0
    for i in range(k):
        n_tmp = int(np.rint(n * d[i]))
        w_v  = w_f(n_tmp, t)
        s += c[i] * w_v
    return s

def check_widder():
    t = 0.5
    n = 30
    k = 100
    res = []
    for i in range(1, n):
        w = w_f(i, t)
        w_acc = w_improve(i, k, t)
        res.append((i, w, w_acc))
    return res

from tabulate import tabulate
widder = check_widder()
print(tabulate(widder, headers=['n', 'Обычный', 'С ускорением']))

y_true = 3.68678
plt.xlabel('n')
plt.ylabel('y')
x = [it[0] for it in widder]
y = [it[1] for it in widder]
y1 = [it[2] for it in widder]
plt.plot(x, y, 'b--',label='Обычный')
plt.plot(x, y1, 'g-.', label='С ускорением')
plt.plot(x, [y_true] * len(x), 'y--', label=f'y={y_true}')
plt.grid(True)
plt.legend()
plt.show()

def series(n: int, t: float) -> float:
    s = 0
    for i in range(n):
        s += (t ** (-5/6 + i/2) / gamma(1/6 + i/2))
    return s

res = []
for i in range(1, 30):
    res.append((i, series(i, t)))
print(tabulate(res, headers=['n', 'Частичная сумма']))

plt.xlabel('n')
plt.ylabel('y')
x = [it[0] for it in res]
y = [it[1] for it in res]
plt.plot(x, y, 'b--',label='Частичная сумма')
plt.plot(x, [y_true] * len(x), 'y--', label=f'y={y_true}')
plt.grid(True)
plt.legend()
plt.show()

