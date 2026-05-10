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
from datetime import datetime


class Experimento:
    def __init__(self):
        pass

    def preparar_dataset(self) -> None:
        """
        Função que modifica a hierarquia do dataset BIPEDv2 que originalmente possui divisões em Treino e Teste
        para o gabarito e para imagem normal. A modificação feita é juntar as partições de treino e teste de
        imagens originais em uma só (imagens) e para imagens já detectadas (gabarito).
        """

        def mover_arquivos(fontes: list[str], destino: Path) -> None:
            contador = 0
            for fonte in fontes:
                p_fonte = Path(fonte)
                if not p_fonte.exists():
                    print(f"Pasta {fonte} não encontrada.")
                    continue
                for arquivo in p_fonte.glob("*.*"):
                    if arquivo.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        shutil.copy2(arquivo, destino / arquivo.name)
                        contador += 1
            print(f"{contador} imagens movidas para {destino}/")

        origens_imgs = ["BIPEDv2/BIPED/edges/imgs/train/rgbr/real", "BIPEDv2/BIPED/edges/imgs/test/rgbr"]
        origens_gabs = ["BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real", "BIPEDv2/BIPED/edges/edge_maps/test/rgbr"]

        pasta_final_imagens    = Path("BIPEDv2_Adaptado/imagens")
        pasta_final_gabarito   = Path("BIPEDv2_Adaptado/gabarito")
        pasta_final_detectados = Path("BIPEDv2_Adaptado/detectados")

        pasta_final_imagens.mkdir(parents=True, exist_ok=True)
        pasta_final_gabarito.mkdir(parents=True, exist_ok=True)
        pasta_final_detectados.mkdir(parents=True, exist_ok=True)

        mover_arquivos(origens_imgs, pasta_final_imagens)
        mover_arquivos(origens_gabs, pasta_final_gabarito)

    def salvar_resultados_mini_estudo(self, resultados: list[dict], nome_arq: str = "resultados.csv") -> None:
        """
        Recebe uma lista dos resultados do experimento e salva em CSV.
        """
        cabecalho = resultados[0].keys()
        with open(nome_arq, 'w', newline='', encoding='utf-8') as f:
            escritor = csv.DictWriter(f, fieldnames=cabecalho)
            escritor.writeheader()
            escritor.writerows(resultados)
        print(f"\nResultados salvos em: {nome_arq}\n")

    def aplicar_deteccao(self, img: cv2.typing.MatLike, filtro: Callable, **kwargs) -> cv2.typing.MatLike:
        return filtro(img, **kwargs)

    def calcular_metricas(self, img: cv2.typing.MatLike, img_gabarito: cv2.typing.MatLike) -> List[float]:
        """
        Compara pixel a pixel e calcula F1-Score, Precision e Recall.
        Retorna [f1_score, precision, recall].
        """
        LIMIAR = 127

        borda_predita = (img >= LIMIAR)
        borda_real    = (img_gabarito >= LIMIAR)

        verdadeiro_positivo = np.sum(borda_predita & borda_real)
        falso_positivo      = np.sum(borda_predita & ~borda_real)
        falso_negativo      = np.sum(~borda_predita & borda_real)

        recall    = verdadeiro_positivo / (verdadeiro_positivo + falso_negativo) if (verdadeiro_positivo + falso_negativo) > 0 else 0.0
        precision = verdadeiro_positivo / (verdadeiro_positivo + falso_positivo) if (verdadeiro_positivo + falso_positivo) > 0 else 0.0
        f1_score  = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return [f1_score, precision, recall]

    def mini_estudo01(self, filtros: dict, num_repeticoes: int) -> None:
        """
        Percorre todas as imagens do dataset BIPEDv2, aplica cada filtro e coleta:
            - tempo_ms:    tempo real decorrido (wall clock)
            - cpu_time_ms: tempo que a CPU gastou executando o processo
        """
        lista_resultados = []
        pasta_imagens = Path("BIPEDv2_Adaptado/imagens")
        arquivos = sorted(list(pasta_imagens.glob("*.jpg")))

        for repeticao in range(1, num_repeticoes + 1):
            print(f"\nExecução número: {repeticao}")

            for caminho_img in arquivos:
                img = cv2.imread(str(caminho_img), cv2.IMREAD_GRAYSCALE)
                print(f"    Imagem atual: {caminho_img}")

                if img is None:
                    continue

                for chave, valor in filtros.items():
                    funcao     = valor["func"]
                    parametros = valor["params"]
                    
                    if(chave == "Sobel"): # Se for Sobel, o jeito de aplicar é diferente
                            if(parametros["dx"] == 0):
                                parametros2 = parametros.copy()
                                parametros2["dx"] = 1
                                parametros2["dy"] = 0
                            
                            else:
                                parametros2 = parametros.copy()
                                parametros2["dy"] = 1
                                parametros2["dx"] = 0

                    print(f"        - Função: {chave}")

                    # ── Horário de início ─────────────────────────────────────
                    inicio = datetime.now()
                    data_formatada_inicio = inicio.strftime("%d/%m/%Y %H:%M:%S")

                    # ── Medição ───────────────────────────────────────────────
                    tempo_wall_inicio = time.perf_counter()   # tempo real
                    tempo_cpu_inicio  = time.process_time()   # tempo de CPU

                    # ── Detecção ──────────────────────────────────────────────
                    if(funcao == "chave"): # Se for Sobel tem que executar desse jeito
                        resultado_x = self.aplicar_deteccao(img, funcao, **parametros)  # Horizontal edges
                        resultado_y = self.aplicar_deteccao(img, funcao, **parametros2)  # Vertical edges
                        
                        # Compute gradient magnitude
                        mag_grad = cv2.magnitude(resultado_x, resultado_y)
                        resultado_borda = cv2.convertScaleAbs(mag_grad)
                    else:
                        resultado_borda = self.aplicar_deteccao(img, funcao, **parametros)

                    # ── Coleta pós-execução ───────────────────────────────────
                    tempo_wall_fim = time.perf_counter()
                    tempo_cpu_fim  = time.process_time()

                    tempo_ms    = (tempo_wall_fim - tempo_wall_inicio) * 1000
                    cpu_time_ms = (tempo_cpu_fim  - tempo_cpu_inicio)  * 1000

                    # ── Salvar imagem detectada e carregar gabarito ───────────
                    caminho_imagem         = str(caminho_img)
                    caminho_dest_detectada = (
                        f"{caminho_imagem.replace('imagens', 'detectados').split('.')[0]}"
                        f"_{chave}.jpg"
                    )
                    caminho_gabarito = (
                        f"{caminho_imagem.replace('imagens', 'gabarito').split('.')[0]}.png"
                    )

                    cv2.imwrite(caminho_dest_detectada, resultado_borda)
                    img_gabarito = cv2.imread(caminho_gabarito, cv2.IMREAD_GRAYSCALE)

                    metricas = self.calcular_metricas(resultado_borda, img_gabarito)

                    fim = datetime.now()
                    data_formatada_fim = fim.strftime("%d/%m/%Y %H:%M:%S")

                    # ── Registrar resultado ───────────────────────────────────
                    lista_resultados.append({
                        "repeticao":      repeticao,
                        "nome_img":       caminho_img.name,
                        "filtro":         chave,
                        "config":         parametros,
                        "f1score":        metricas[0],
                        "precision":      metricas[1],
                        "recall":         metricas[2],
                        "tempo_ms":       tempo_ms,
                        "cpu_time_ms":    cpu_time_ms,
                        "horario_inicio": data_formatada_inicio,
                        "horario_fim":    data_formatada_fim,
                    })

        self.salvar_resultados_mini_estudo(lista_resultados)
        self.gerar_grafico_mini_estudo_01("resultados.csv")

    def gerar_grafico_mini_estudo_01(self, caminhos_resultados: str) -> None:
        df_completo = pd.read_csv(caminhos_resultados)

        relatorio_final = df_completo.groupby('filtro').agg({
            'f1score':     ['mean', 'std'],
            'tempo_ms':    ['mean', 'std'],
            'cpu_time_ms': ['mean', 'std'],
        }).reset_index()

        relatorio_final.columns = [
            'filtro',
            'f1_medio',    'f1_std',
            'tempo_medio', 'tempo_std',
            'cpu_medio',   'cpu_std',
        ]

        print(relatorio_final)
