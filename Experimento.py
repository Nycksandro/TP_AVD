from typing import List, Callable
import shutil
from pathlib import Path
import cv2
import csv


class Experimento:
    def __init__(self):
        pass

    def preparar_dataset(self) -> None:
        """
        Função que modifica a hierarquia do dataset BIPEDv2 que originalmente possui divisões em Treino e Teste para o gabarito e para imagem normal.
        A modificação feita é juntar as partição de treino e teste de imagens originais eu uma só (imagens) e para imagens já detectadas (gabarito).  
        """

        def mover_arquivos(fontes: list[str], destino: Path) -> None:
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

    def organizar_tarefa_07(self) -> None:
        """
        Função para separar uma amostra para a Tarefa 01 (13/04) de AVD. A amostra corresponde ao conjunto de teste do BIPEDv2 original, composto por 50 imagens
        """
       
        caminho_origem = Path("BIPEDv2/BIPED/edges/imgs/test/rgbr")
        caminho_destino = Path("BIPEDv2_Adaptado/tarefa01")
        caminho_destino_originais = Path("BIPEDv2_Adaptado/tarefa01/originais")
        caminho_destino_detectados = Path("BIPEDv2_Adaptado/tarefa01/detectados")

        # Criar a pasta de destino (parents=True cria toda a árvore se necessário)
        caminho_destino.mkdir(parents=True, exist_ok=True)
        caminho_destino_originais.mkdir(parents=True, exist_ok=True)
        caminho_destino_detectados.mkdir(parents=True, exist_ok=True)

        print(f"Iniciando cópia de: {caminho_origem}")

        # Verificar se a origem existe para evitar erros
        if not caminho_origem.exists():
            print(f"Erro: O caminho de origem '{caminho_origem}' não foi encontrado.")
            return

        # Percorrer e copiar
        contador = 0
        # Buscamos por extensões comuns de imagem (maiúsculas ou minúsculas)
        extensoes = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        
        for extensao in extensoes:
            for arquivo in caminho_origem.glob(extensao):
                # shutil.copy2 mantém os metadados originais do arquivo (como data de criação)
                shutil.copy2(arquivo, caminho_destino_originais / arquivo.name)
                contador += 1

        print(f" Concluído! {contador} imagens foram copiadas para {caminho_destino_originais}")

    def salvar_resultados_tarefa_07(self, resultados: list[dict[str, any]], nome_arq: str = "resultados.csv") ->None:
        """
        Função que recebe uma lista dos resultados do experimento da tarefa 07 e transforma em um CSV contendo todas as informações de cada execução de algoritmo
        """
        # A lista de resultados deve ser assim: 
        # resultados = [{'arquivo': 'img1.jpg', 'filtro': 'Sobel', 'tempo_ms': 15.4}, ...]

        nome_arquivo = nome_arq

        # Pegamos as chaves do primeiro dicionário para serem o cabeçalho
        cabecalho = resultados[0].keys()
        with open(nome_arquivo, 'w', newline='', encoding='utf-8') as f:

            escritor = csv.DictWriter(f, fieldnames=cabecalho)
            
            escritor.writeheader()  # Escreve o nome das colunas
            escritor.writerows(resultados) # Escreve todas as linhas da lista

        print(f"Resultados salvos em {nome_arquivo}")

    def aplicar_deteccao(self, img:cv2.typing.MatLike, filtro: Callable[..., cv2.typing.MatLike], **kwargs) -> cv2.typing.MatLike:
        """
        Recebe uma imagem e o método/operador desejado para aplicar na imagem
        """
        return filtro(img, **kwargs) # Aplica o filtro passando os parâmetros

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