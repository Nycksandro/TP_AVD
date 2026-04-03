import os
import cv2
from skimage import filters
import time
from Experimento import Experimento

# Operadores e métodos já implementados (e conhecidos)

# cv2.Sobel() # Sobel
# cv2.Laplacian Ou filters.laplace()
# feature.canny() # Canny

# egde_laplace = filters.laplace() # Laplace
# edge_roberts = filters.roberts(image) # Roberts
# edge_sobel   = filters.sobel(image) # Sobel
# edge_scharr  = filters.scharr(image) # Scharr
# edge_canny   = feature.canny(image, sigma=3) # Canny


# threshold1 (Limiar Inferior): Pixels abaixo deste valor são rejeitados.
# threshold2 (Limiar Superior): Pixels acima deste valor são considerados bordas fortes.
#imagem_saida = feature.canny(imagem) #

#cv2.imwrite(caminho_saida, imagem_saida)
# import matplotlib.pyplot as plt
# from skimage import feature, data, color

# # 1. Carregar uma imagem de exemplo e converter para tons de cinza (obrigatório para Canny)
# img = data.camera() 

# # 2. Aplicar o algoritmo de Canny
# # O parâmetro 'sigma' controla a suavização (quanto maior, menos ruído)
# edges = feature.canny(img, sigma=1.5)
# # 3. Mostrar o resultado lado a lado
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(imagem, cmap='gray')
# ax[0].set_title('Imagem Original')
# ax[0].axis('off')

# ax[1].imshow(imagem_saida, cmap='gray')
# ax[1].set_title('Detecção de Bordas (Canny)')
# ax[1].axis('off')

# plt.show()

caminho_normal = "BIPEDv2_Adaptado/imagens/RGB_001.jpg"
caminho_gabarito = "BIPEDv2_Adaptado/gabarito/RGB_001.png"

exp = Experimento()
exp.preparar_dataset() # Prepara o dataset

imagem_normal = cv2.imread(caminho_normal, cv2.IMREAD_GRAYSCALE)
imagem_gabarito = cv2.imread(caminho_gabarito, cv2.IMREAD_GRAYSCALE)
imagem_detectada = cv2.Canny(imagem_normal, 60, 170)
cv2.imwrite('Testes.png',imagem_detectada)

metricas = exp.calcular_metricas(imagem_detectada, imagem_gabarito)
print(f"F1-Score: {metricas[0]}\n")
print(f"Precision: {metricas[1]}\n")
print(f"Recall: {metricas[2]}\n")