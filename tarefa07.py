from Experimento import Experimento
from pathlib import Path
import cv2
import time

exp = Experimento()
exp.organizar_tarefa_07() # Separando a amostra

caminho_pasta_originais = Path("BIPEDv2_Adaptado/tarefa01/originais") # Onde está localizado as amostras

resultados = []

arquivos = sorted(list(caminho_pasta_originais.glob("*.jpg"))) # Pegamos todos os arquivos de imagem 

filtros = { "Sobel": { # Referência das funções e os seus parâmetros
        "func": cv2.Sobel, 
        "params": {"ddepth": cv2.CV_8U, "dx": 1, "dy": 0, "ksize": 3}
        },
        "Laplacian": {
            "func": cv2.Laplacian, 
            "params": {"ddepth": cv2.CV_8U, "ksize": 3}
        }
}

print(f"Iniciando Experimento com {len(arquivos)} imagens...")

for caminho_img in arquivos:
    img = cv2.imread(str(caminho_img)) # Carrega a imagem
    altura = img.shape[0]
    largura = img.shape[1]
    if (img is not None): # Verifica se ela existe
        for chave,valor in filtros.items(): # Aplica na imagem cada filtro escolhido
            funcao = valor["func"]   
            parametros = valor["params"] 
            
            inicio = time.perf_counter() # Começa a contar o tempo
            resultado_borda = exp.aplicar_deteccao(img, funcao, **parametros) # Aplica o filtro desejado
            
            fim = time.perf_counter() # Encerra a contagem do tempo
            
            tempo_ms = (fim - inicio) * 1000
            resultados.append({ # Armazenar dados
                "nome": caminho_img.name,
                "filtro": chave,
                "tempo_ms": tempo_ms,
                "resolucao": f"{largura}x{altura}"
            })
            caminho_imagem = str(caminho_img)    
            caminho_destino = f"{caminho_imagem.replace("originais", "detectados").split(".")[0]}_{chave}.jpg"
            cv2.imwrite(caminho_destino, resultado_borda)

exp.salvar_resultados_tarefa_07(resultados)