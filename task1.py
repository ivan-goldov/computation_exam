import numpy as np
import tabulate
from typing import List, Any

def create_a(size: int) -> np.array:
    a = []
    for i in range(1, size+1):
        a.append([])
        for j in range(1, size+1):
            b_k = (size + 1 - j) 
            b_k **= -2
            a_k = 2 * i
            a[-1].append(a_k ** b_k)
    return np.array(a)

def create_z(size: int) -> np.array:
    return np.ones(size)

def calculate_u(a: np.array, z: np.array) -> np.array:
    return a @ z

def calculate_b(a: np.array) -> np.array:
    values, matrix = np.linalg.eig(a)
    diagonal_matrix = np.diag(np.sqrt(values))
    inverse_matrix = np.linalg.inv(matrix)
    return  matrix @ diagonal_matrix @ inverse_matrix

def print_eig(a: np.array):
    values, matrix = np.linalg.eig(a)
    print(f'Матрица\n{a}')
    print(f'Собственные числа\n{values}')
    print(f'Все собственные числа матрицы положительны? {"Да" if all(values > 0) else "НЕТ!"}')
    print(f'Собственные векторы (строки)\n {matrix.T}')

def print_table(max_n: int = 10):
    print('Заполняем таблицу')
    table = []
    for n in range(2, max_n):
        a = create_a(n)
        b = calculate_b(a)
        a_cond = np.linalg.cond(a)
        b_cond = np.linalg.cond(b)
        norm = np.linalg.norm(a - b @ b)
        table.append((n, a_cond, b_cond, norm))
        # print(f'n = {n}, cond(A)= {a_cond}, cond(B) = {b_cond}, ||A-B^2|| = {norm}')
    return table

def compare_regularizations(max_n: int = 10) -> List[List[Any]]:
    n_array = [n for n in range(2, max_n)]
    alpha_array = [10 ** (-13 + i) for i in range(13)]
    table = []
    for n in n_array:
        for alpha in alpha_array:
            a = create_a(n)
            z = create_z(n)
            u = calculate_u(a, z)
            a_hermite = a.T
            new_u = a_hermite @ u
            new_matrix = a_hermite @ a + np.eye(n) * alpha
            new_z = np.linalg.solve(new_matrix, new_u)
            norm = np.linalg.norm(z - new_z)
            b = calculate_b(a)
            b_hermite = b.T
            new_b = b_hermite @ b + np.eye(n) * alpha
            new_u_b = b_hermite @ (np.linalg.inv(b) @ u)
            new_z_b = np.linalg.solve(new_b, new_u_b)
            norm2 = np.linalg.norm(new_z_b - z)
            table.append([n, alpha, norm, norm2])
    return table

a = create_a(5)
print(a)
z = create_z(5)
print(z.reshape(5, 1))
print((a @ z).reshape(5, 1))
b = calculate_b(a)
print(b)
print(b @ b)
print_eig(a)

table = print_table()
print(tabulate.tabulate(table, headers=['n', 'cond(A)', 'cond(B)', '||A-B^2||'], tablefmt='fancy_grid'))

table = compare_regularizations()
print(tabulate.tabulate(table, headers=['n', 'alpha', 'Первый метод',' Второй метод'], tablefmt='fancy_grid'))

