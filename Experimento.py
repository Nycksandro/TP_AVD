from typing import List, Callable
import shutil
import psutil
import os
from pathlib import Path
import cv2
import time
import csv
import numpy as np
import pandas as pd


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
        pasta_final_detectados = Path("BIPEDv2_Adaptado/detectados")


        # Cria pastas de destino se elas não existirem
        pasta_final_imagens.mkdir(parents=True, exist_ok=True)
        pasta_final_gabarito.mkdir(parents=True, exist_ok=True)
        pasta_final_detectados.mkdir(parents=True, exist_ok=True)

        # Move os arquivos
        mover_arquivos(origens_imgs, pasta_final_imagens)
        mover_arquivos(origens_gabs, pasta_final_gabarito)

    def salvar_resultados_mini_estudo(self, resultados: list[dict[str, any]], nome_arq: str = "resultados.csv") ->None:
        """
        Função que recebe uma lista dos resultados do experimento da tarefa 07 e transforma em um CSV contendo todas as informações de cada execução de algoritmo
        """
        # A lista de resultados deve ser assim: 
        # resultados = [{"repeticao": repeticao, "nome_img": caminho_img.name, "filtro": chave, "config": valor, "f1score": metricas[0], "precision": metricas[1], "recall": metricas[2], "tempo_ms": tempo_ms, "memoria_delta_mb": consumo_ram, "cpu_time_sec": tempo_cpu_gasto}, ...]

        nome_arquivo = nome_arq

        # Pegamos as chaves do primeiro dicionário para serem o cabeçalho
        cabecalho = resultados[0].keys()
        with open(nome_arquivo, 'w', newline='', encoding='utf-8') as f:

            escritor = csv.DictWriter(f, fieldnames=cabecalho)
            
            escritor.writeheader()  # Escreve o nome das colunas
            escritor.writerows(resultados) # Escreve todas as linhas da lista

        print(f"\nResultados salvos em: {nome_arquivo}\n")

    def aplicar_deteccao(self, img:cv2.typing.MatLike, filtro: Callable[..., cv2.typing.MatLike], **kwargs) -> cv2.typing.MatLike:
        """
        Recebe uma imagem e o método/operador desejado para aplicar na imagem (juntamente com)
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
        
        LIMIAR = 127 # Se for menor que 128 consideramos PRETO, se for maior consideramos BRANCO
        falso_positivo = 0
        falso_negativo = 0
        verdadeiro_positivo = 0 
        verdadeiro_negativo = 0
        
        borda_predita = (img >= LIMIAR)
        borda_real = (img_gabarito >= LIMIAR)

        # Usando lógica boleana
        verdadeiro_positivo = np.sum(borda_predita & borda_real) # Preveu Borda e Realmente era Borda (TP)
        verdadeiro_negativo = np.sum(~borda_predita & ~borda_real) # Detectou não borda e realmente não era borda (TN)
        falso_positivo = np.sum(borda_predita & ~borda_real) # Detectou borda e não era Borda (FP)
        falso_negativo= np.sum(~borda_predita & borda_real) # Detectou não borda e era Borda (FN)

        recall = verdadeiro_positivo/(verdadeiro_positivo + falso_negativo)#  R = TP / (TP + FN)
        precision = verdadeiro_positivo/(verdadeiro_positivo + falso_positivo)# P = TP / (TP + FP) 
        f1_score = 2*(precision*recall)/(precision+recall) # 2 * (P * R)  (P + R)
        return [f1_score, precision, recall]

    def mini_estudo01(self, filtros, num_repeticoes: int):
        """
        A carga do Mini Estudo 01 é todo o dataset BIPEDv2, ou seja, utilizar as 250 imagens do dataset, onde junto as imagens da partição de Treino e Teste já que não iremos aplicar técnicas de ML, e sim algoritmos de detecção de borda.
        A ideia é percorrer e coletar as informações (antes e depois) da execução de cada algoritmo de detecção sobre cada imagem do dataset e registrar:
            - Tempo de execução
            - Consumo de RAM médio
            - Consumo de CPU médio
        
        """
        
        lista_resultados = [] # Pra guardar todos os resultados
        pasta_imagens = Path("BIPEDv2_Adaptado/imagens")

        arquivos = sorted(list(pasta_imagens.glob("*.jpg"))) # Pegamos todos os arquivos de imagem 

        for repeticao in range(1, num_repeticoes+1):
            print(f"\nExecução número: {repeticao}")
            for caminho_img in arquivos:
                img = cv2.imread(str(caminho_img)) # Carrega a imagem
                print(f"    Imagem atual: {caminho_img}")

                if (img is not None): # Verifica se ela existe
                    for chave,valor in filtros.items(): # Aplica na imagem cada filtro escolhido
                        funcao = valor["func"]   # Obtem qual é a função
                        parametros = valor["params"] # Obtêm os parâmetros da função 

                        print(f"        - Função: {chave}")

                        # Coleta pré-execução

                        processo = psutil.Process(os.getpid()) # # Inicializa o monitor do processo atual
                        memoria_antes = processo.memory_info().rss / (1024 * 1024) # MB
                    
                        cpu_antes = processo.cpu_times().user + processo.cpu_times().system # O cpu_percent precisa de um pequeno intervalo ou do acumulado do processo
                        tempo_inicio = time.perf_counter()

                        # Aplicação da deteccção
                        resultado_borda = self.aplicar_deteccao(img, funcao, **parametros)
                        
                        # Coleta pós-execução

                        tempo_fim = time.perf_counter()
                        cpu_depois = processo.cpu_times().user + processo.cpu_times().system
                        memoria_depois = processo.memory_info().rss / (1024 * 1024) # MB

                        # Calculando tempo gasto, ram e cpu consumida

                        tempo_ms = (tempo_fim - tempo_inicio) * 1000 # Tempo
                        consumo_ram = memoria_depois - memoria_antes # RAM
                        tempo_cpu_gasto = cpu_depois - cpu_antes # CPU

                        # Essas 5 linhas abaixo serve pra salvar a imagem detectada e ler a correspondente ao gabarito 
                        caminho_imagem = str(caminho_img)    
                        caminho_destino_img_detectada = f"{caminho_imagem.replace("imagens", "detectados").split(".")[0]}_{chave}.jpg"
                        caminho_gabarito = f"{caminho_imagem.replace("imagens", "gabarito").split(".")[0]}.png"
                        
                        cv2.imwrite(caminho_destino_img_detectada, resultado_borda) # Registrar a imagem detectada na variavel "resultado_borda"
                        img_gabarito = cv2.imread(str(caminho_gabarito)) # Lendo a imagem gabarito

                        metricas = self.calcular_metricas(resultado_borda, img_gabarito)

                        # Registrar na lista de resultados
                        lista_resultados.append({
                            "repeticao": repeticao,
                            "nome_img": caminho_img.name,
                            "filtro": chave,
                            "config": parametros,
                            "f1score": metricas[0],
                            "precision": metricas[1],
                            "recall": metricas[2],
                            "tempo_ms": tempo_ms,
                            "memoria_delta_mb": consumo_ram,
                            "cpu_time_sec": tempo_cpu_gasto
                        })
        self.salvar_resultados_mini_estudo(lista_resultados)
        self.gerar_grafico_mini_estudo_01("resultados.csv")


    def gerar_grafico_mini_estudo_01(self, caminhos_resultados):
        # 1. Criar o DataFrame com todos os resultados do dataset
        df_completo = pd.read_csv(caminhos_resultados)

        # 2. Agrupar por filtro para ter a visão global
        relatorio_final = df_completo.groupby('filtro').agg({
            'f1score': ['mean', 'std'],
            'tempo_ms': ['mean', 'std']
        }).reset_index()

        # Limpar as colunas para facilitar o uso
        relatorio_final.columns = ['filtro', 'f1_medio', 'f1_std', 'tempo_medio', 'tempo_std']

        print(relatorio_final)