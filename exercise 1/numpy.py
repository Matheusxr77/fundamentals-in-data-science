"""
Parte 1: Numpy
NumPy: práticas necessárias com uso da biblioteca NumPy
!pip install numpy
"""

# Importando a biblioteca NumPy
import numpy as np

# 1) Crie 1 array com 3 elementos distintos
array_1 = np.array([5, 10, 15])
print(array_1)

# 2) Crie 1 array de zeros
array_zeros = np.zeros(3)
print(array_zeros)

# 3) Crie 1 array de uns
array_uns = np.ones(3)
print(array_uns)

# 4) Crie 1 array de 4 elementos arbitrários
array_arbitrary = np.array([2, 4, 6, 8])
print(array_arbitrary)

# 5) Crie 1 array a partir de um intervalo de números (sequência)
array_sequence = np.arange(10)
print(array_sequence)

# 6) Crie 1 array a partir de um intervalo de números e que apresente uma sequência de 3 em 3.
array_sequence_3 = np.arange(0, 10, 3)
print(array_sequence_3)

# 7) Explique o resultado desse código: np.linspace(0,22,5)
array_linspace = np.linspace(0, 22, 5)
print(array_linspace)

# 8) Crie um array de 21 elementos e exiba:
array_21 = np.arange(21)
print(array_21)

# 8.1) Tipo
print(array_21.dtype)

# 8.2) Número de elementos
print(array_21.size)

# 8.3) Consumo de bytes por elemento
print(array_21.itemsize)

# 8.4) Número de elementos
print(array_21.nbytes)

# 8.5) Número de dimensões
print(array_21.ndim)

# 9) Criar uma lista A de 3 listas com 3 elementos
lista_A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(lista_A)

# 10) Copiar a lista do item anterior para um array multidimensional chamado multi_B
multi_B = lista_A.copy()
print(multi_B)

# 11) Exibir o número de elementos em cada dimensão do array lista A e B
print(lista_A.shape)
print(multi_B.shape)

# 12) Usando o comando reshape crie transforme uma lista de 12 elementos em 1 array (3,4)
array_12 = np.arange(12).reshape(3, 4)
print(array_12)

# 13) Usando o comando reshape transforme o array anterior em uma lista de 12 elementos
array_12.reshape(12)
print(array_12)

# 14) Usando o comando reshape transforme a lista anterior em um array (2,2,3)
array_14 = array_12.reshape(2, 2, 3)
print(array_14)

# 15) Crie um array de 21 elementos e apresente:
array_21 = np.arange(21)
print(array_21)

# 15.1) O elemento 3
print(array_21[3])

# 15.2) O último elemento
print(array_21[-1])

# 15.3) Os elementos no intervalo de 3 ao 9
print(array_21[3:10])

# 16) Crie 1 array de 21 elementos usando o comando arange e reshape em uma mesma linha de comando e faça:
array_21 = np.arange(21).reshape(3, 7)
print(array_21)

# 16.1) Apresente o elemento 2 (linha 2)
print(array_21[1])

# 16.2) Apresente o elemento 3 da linha 2
print(array_21[2, 3])

# 16.3) Apresente as linhas 0 e 1 e seus elementos
print(array_21[:2])

# 16.4) Apresente os elementos da coluna 3 de cada linha do array
print(array_21[:, 3])

# 16.5) Apresente apenas os 3 últimos elementos da primeira e segunda linha do array
print(array_21[:2, -3:])

# 16.6) Apresente apenas os 5 últimos elementos de cada linha do array
print(array_21[:, -5:])

# 17) Crie 1 array de 21 elementos usando o comando arange e reshape em uma mesma linha de comando, depois de criado altere o primeiro elemento para 51
array_21 = np.arange(21).reshape(3, 7)
array_21[0, 0] = 51
print(array_21)

# 18) No array anterior (3,7) altere apenas as linhas 2 e 3 para que os 3 primeiros elementos de cada seja 0.
array_21[2:, :3] = 0
print(array_21)

# 19) Crie duas listas e apresente em uma nova lista:
lista_1 = [1, 2, 3]
lista_2 = [4, 5, 6]
lista_3 = np.array([lista_1, lista_2])
print(lista_3)

# 19.1) A soma os elementos da lista
print(lista_3.sum())

# 19.2) A multiplicação dos elementos dessas listas
print(lista_3.prod())

# 19.3) A divisão dos elementos dessas listas
print(lista_3.cumsum())

# 20) Crie 1 array de 24 elementos e apresente:
array_24 = np.arange(24)
print(array_24)

# 20.1) O maior valor
print(array_24.max())

# 20.2) O menor valor
print(array_24.min())

# 20.3) A soma dos valores
print(array_24.sum())

# 20.4) A média dos valores
print(array_24.mean())

# 20.5) A soma de cada linha
print(array_24.sum(axis=1))

# 20.6) A soma de cada coluna
print(array_24.sum(axis=0))

# 20.7) A soma de cada coluna
print(array_24.sum(axis=0))

# 21) Crie na forma de polinômio usando o comando poly1d os seguintes polinômios:
# 21.1) 3x + 4
print(np.poly1d([3, 4]))

# 21.2) 4x^3 + 3x^2 + 2x + 1 
print(np.poly1d([4, 3, 2, 1]))

# 21.3) 2x^3 + 3
print(np.poly1d([2, 0, 0, 3]))

# 21.4) x^2 + 2x + 3
print(np.poly1d([0, 1, 2, 3]))

# 21.5) Multiplique o polinômios c e d
c = np.poly1d([2, 0, 0, 3])
d = np.poly1d([0, 1, 2, 3])
print(c * d)

# 21.6) Derive o polinômio d
print(d.deriv())

# 21.7) Integre o polinômio d
print(d.integ())