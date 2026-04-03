from typing import List, Callable
import shutil
from pathlib import Path
import cv2

class Experimento:
    def __init__(self):
        pass

    def preparar_dataset(self):
        """
        Função que modifica a hierarquia do dataset BIPEDv2 que originalmente possui divisões em Treino e Teste para o gabarito e para imagem normal.
        A modificação feita é juntar as partição de treino e teste de imagens originais eu uma só (imagens) e para imagens já detectadas (gabarito).  
        """

        def mover_arquivos(fontes: list[str], destino: Path):
            """
            Função para mover as imagens de uma fonte para o destino
            """
            contador = 0
            for fonte in fontes:
                p_fonte = Path(fonte)
                if not p_fonte.exists():
                    print(f"Pasta {fonte} não encontrada.")
                    continue
                    
                # Percorre as imagem
                for arquivo in p_fonte.glob("*.*"):
                    if arquivo.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        shutil.copy2(arquivo, destino / arquivo.name)
                        contador += 1
            
            print(f"{contador} imagens movidas para {destino}/")

        origens_imgs = ["BIPEDv2/BIPED/edges/imgs/train/rgbr/real", "BIPEDv2/BIPED/edges/imgs/test/rgbr"]
        origens_gabs = ["BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real", "BIPEDv2/BIPED/edges/edge_maps/test/rgbr"]

        # Define pastas de destino
        pasta_final_imagens = Path("BIPEDv2_Adaptado/imagens")
        pasta_final_gabarito = Path("BIPEDv2_Adaptado/gabarito")

        # Cria pastas de destino se elas não existirem
        pasta_final_imagens.mkdir(parents=True, exist_ok=True)
        pasta_final_gabarito.mkdir(parents=True, exist_ok=True)

        # Move os arquivos
        mover_arquivos(origens_imgs, pasta_final_imagens)
        mover_arquivos(origens_gabs, pasta_final_gabarito)

    def aplicar_deteccao(self, img:cv2.typing.MatLike, filtro: Callable[..., cv2.typing.MatLike]):
        """
        Recebe uma imagem e o método/operador desejado para aplicar na imagem
        """
        return

    def calcular_metricas(self, img: cv2.typing.MatLike, img_gabarito: cv2.typing.MatLike) -> List[float]:
        """
        Percorre ambas as imagens e compara pixel a pixel e classifica se foi Verdadeiro Positivo, Verdadeiro Negativo, Falso Positivo e Falso Negativo.
        No fim, calcula-se as métricas Precision e Recall, e por fim a F1-Score.

        Entrada:
            img: Imagem detectada
            img_gabarito: Imagem de referência

        Saída: Retorna um conjunto de métricas  
            [F1-Score, Precision, Recall]  

        """
        altura, largura = img.shape[:2] # Pegando as dimensões da imagem
        
        PRETO = 50
        BRANCO = 200
        falso_positivo = 0
        falso_negativo = 0
        verdadeiro_positivo = 0 
        verdadeiro_negativo = 0
        
        for i in range(altura):
            for j in range(largura):
                pixel_img = img[i][j]
                pixel_gabarito = img_gabarito[i][j]

                if(pixel_img <= PRETO and pixel_gabarito <= PRETO): # Se não era borda e não detectou (Verdadeiro Negativo)
                    verdadeiro_negativo +=1
                elif(pixel_img >= BRANCO and pixel_gabarito >= BRANCO): # Se era borda e detectou borda (Verdadeiro positivo)
                    verdadeiro_positivo +=1
                elif(pixel_img <= PRETO and pixel_gabarito >= BRANCO): # Se era borda e não detectou (Falso Negativo)
                    falso_negativo +=1
                elif(pixel_img >= BRANCO and pixel_gabarito <= PRETO): # Se não era borda e detectou borda (Falso Positivo)
                    falso_positivo +=1

        recall = verdadeiro_positivo/(verdadeiro_positivo + falso_negativo)#  R = TP / (TP + FN)
        precision = verdadeiro_positivo/(verdadeiro_positivo + falso_positivo)# P = TP / (TP + FP) 
        f1_score = 2*(precision*recall)/(precision+recall) # 2 * (P * R)  (P + R)
        return [f1_score, precision, recall]