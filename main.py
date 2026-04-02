import os
import cv2
from skimage import filters, feature

# cv2.Sobel() # Sobel
# cv2.Laplacian Ou filters.laplace()
# feature.canny() # Canny

# egde_laplace = filters.laplace() # Laplace
# edge_roberts = filters.roberts(image) # Roberts
# edge_sobel   = filters.sobel(image) # Sobel
# edge_scharr  = filters.scharr(image) # Scharr
# edge_canny   = feature.canny(image, sigma=3) # Canny

# Esse main realiza a comparação e aplicação das métricas

caminho_gabarito = "BIPEDv2/BIPED/edges/edge_maps/train"
caminho_imagens = "BIPEDv2/BIPED/edges/imgs/train"

caminho_entrada = "abobora.png"
caminho_saida = "testes.png"

imagem = cv2.imread(caminho_entrada)
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# threshold1 (Limiar Inferior): Pixels abaixo deste valor são rejeitados.
# threshold2 (Limiar Superior): Pixels acima deste valor são considerados bordas fortes.
imagem_saida = feature.canny(imagem) #

#cv2.imwrite(caminho_saida, imagem_saida)
import matplotlib.pyplot as plt
from skimage import feature, data, color

# 1. Carregar uma imagem de exemplo e converter para tons de cinza (obrigatório para Canny)
img = data.camera() 

# 2. Aplicar o algoritmo de Canny
# O parâmetro 'sigma' controla a suavização (quanto maior, menos ruído)
edges = feature.canny(img, sigma=1.5)

# 3. Mostrar o resultado lado a lado
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(imagem, cmap='gray')
ax[0].set_title('Imagem Original')
ax[0].axis('off')

ax[1].imshow(imagem_saida, cmap='gray')
ax[1].set_title('Detecção de Bordas (Canny)')
ax[1].axis('off')

plt.show()