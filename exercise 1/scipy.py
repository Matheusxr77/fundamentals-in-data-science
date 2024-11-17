"""
Parte 2: SciPy
SciPy: práticas necessárias com uso da biblioteca SciPy
!pip install scipy
"""

# Importando a biblioteca SciPy
import scipy

# Usando o submódulo scipy.misc faça:
import scipy.misc as misc
import matplotlib.pyplot as plt

# 1) Carregue e exiba a imagem “face” em seu formato original
face_image = misc.face()

plt.imshow(face_image)
plt.title("Imagem Original")
plt.axis("off")
plt.show()

# 2) Exiba a imagem anterior em escala cinza
import numpy as np

gray_face = np.dot(face_image[..., :3], [0.299, 0.587, 0.114])

plt.imshow(gray_face, cmap='gray')
plt.title("Imagem em Escala de Cinza")
plt.axis('off')
plt.show()

# 3) Apresente o array NumPy referente a imagem.
print("Array NumPy referente a imagem:", gray_face)
print("Dimensões do array:", gray_face.shape)

# Operações de Interpolação de Imagens
# 4) Instale o scikit-image no seu ambiente
# !pip install scikit-image

# 5) Redimensione a imagem gerada anteriormente para 50% do seu tamanho original
from skimage.transform import resize

imagem = misc.face()

print("Dimensões originais:", imagem.shape)

nova_imagem = resize(imagem, (imagem.shape[0] // 2, imagem.shape[1] // 2), anti_aliasing=True)

print("Dimensões após redimensionamento:", nova_imagem.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax = axes.ravel()

ax[0].imshow(imagem, cmap='gray')
ax[0].set_title("Imagem Original")
ax[1].imshow(nova_imagem, cmap='gray')
ax[1].set_title("Imagem Redimensionada (50%)")

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# 6) Volte a imagem para o tamanho original usando interpolação bilinear (Note que mesmo recuperada a imagem não apresenta a mesma qualidade da imagem original.)
imagem_original = misc.face()

imagem_reduzida = resize(imagem_original,
                         (imagem_original.shape[0] // 2, imagem_original.shape[1] // 2),
                         anti_aliasing=True,
                         preserve_range=True).astype(np.uint8)

imagem_recuperada = resize(imagem_reduzida,
                           (imagem_original.shape[0], imagem_original.shape[1]),
                           anti_aliasing=True,
                           order=1,
                           preserve_range=True).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax = axes.ravel()

ax[0].imshow(imagem_original, cmap='gray')
ax[0].set_title("Imagem Original")

ax[1].imshow(imagem_reduzida, cmap='gray')
ax[1].set_title("Imagem Reduzida (50%)")

ax[2].imshow(imagem_recuperada, cmap='gray')
ax[2].set_title("Imagem Recuperada (Tamanho Original)")

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# Usando o submódulo scipy.special faça:
import scipy.special as special

# 7) Calcule e apresente o fatorial do número 4 como exemplo
factorial_4 = special.factorial(4)
print("Fatorial de 4:", factorial_4)

# 8) Apresente os gráficos das funções de Besel e da função de erro Gaussiana
x = np.linspace(0, 10, 100)

bessel_function = special.jn(1, x)
error_function = special.erf(x)

plt.figure(figsize=(10, 5))
plt.plot(x, bessel_function, label='Função de Bessel (J1)')
plt.title("Função de Bessel")
plt.legend()
plt.grid(True)
plt.show()

print("\n")

plt.figure(figsize=(10, 5))
plt.plot(x, error_function, label='Função de Erro Gaussiana')
plt.title("Função de Erro Gaussiana")
plt.legend()
plt.grid(True)
plt.show()

# 9) Apresente o gráfico da função Gama
gamma_function = special.gamma(x)

plt.figure(figsize=(10, 5))
plt.plot(x, gamma_function, label='Função Gama')
plt.title("Função Gama")
plt.legend()
plt.grid(True)
plt.show()

# 10) Apresente os gráficos dos polinômios de Legendre
x = np.linspace(-1, 1, 100)

legendre_polynomials = [special.legendre(i) for i in range(5)]

plt.figure(figsize=(10, 5))
for i, legendre_polynomial in enumerate(legendre_polynomials):
    plt.plot(x, legendre_polynomial(x), label=f'Polinômio de Legendre {i}')
plt.title("Polinômios de Legendre")
plt.legend()
plt.grid(True)
plt.show()

# Usando o submódulo scipy.stats (distribuições discretas e contínuas) faça:
import scipy.stats as stats

# 11) Apresente os gráficos das funções PDF e CDF (CDF é a integral de PDF (Função de Densidade de Probabilidade) e PDF é a derivada de CDF).
x = np.linspace(-4, 4, 100)

normal_distribution = stats.norm(0, 1)

pdf = normal_distribution.pdf(x)
cdf = normal_distribution.cdf(x)

plt.figure(figsize=(10, 5))
plt.plot(x, pdf, label='PDF (Função de Densidade de Probabilidade)')
plt.plot(x, cdf, label='CDF (Função de Distribuição Acumulada)')
plt.title("Distribuição Normal")
plt.legend()
plt.grid(True)
plt.show()

# 12) Apresente o gráfico de probabilidade da função de massa de probabilidade (PMF) para a distribuição binomial (Probabilidade x Número de Sucessos).
n = 10
p = 0.5

binomial_distribution = stats.binom(n, p)

x = np.arange(0, n + 1)

pmf = binomial_distribution.pmf(x)

plt.figure(figsize=(10, 5))
plt.bar(x, pmf, align='center', alpha=0.7)
plt.title("PMF (Probabilidade x Número de Sucessos)")
plt.xlabel("Número de Sucessos")
plt.ylabel("Probabilidade")
plt.grid(True)
plt.show()

# 13) Gere uma amostra aleatória e apresente a estatística t e o valor 
sample = np.random.normal(0, 1, 100)

t_statistic, p_value = stats.ttest_1samp(sample, 0)

print("Estatística t:", t_statistic)
print("Valor p:", p_value)

# 14) Apresente os valores de correlação de Pearson e Spearman e um gráfico com linha de regressão ajustada sobre os pontos de dados.
# 14.1) A correlação de Pearson avalia a relação linear entre duas variáveis contínuas.
# 14.2) A correlação de Spearman avalia a relação monotônica entre duas variáveis contínuas ou ordinais.
x = np.random.rand(100)
y = 2 * x + np.random.normal(0, 0.5, 100)

pearson_corr, _ = stats.pearsonr(x, y)
spearman_corr, _ = stats.spearmanr(x, y)

print("Correlação de Pearson:", pearson_corr)
print("Correlação de Spearman:", spearman_corr)

plt.scatter(x, y, label='Dados')
plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='red', label='Regressão Linear')
plt.title("Correlação de Pearson: {:.2f}\nCorrelação de Spearman: {:.2f}".format(pearson_corr, spearman_corr))
plt.xlabel("Variável X")
plt.ylabel("Variável Y")
plt.legend()
plt.grid(True)
plt.show()