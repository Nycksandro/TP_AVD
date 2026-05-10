## **Experimento de AVD**

Este repositório contém um estudo experimental focado na análise do trade-off entre qualidade de detecção e custo computacional em algoritmos de processamento digital de imagens.

O objetivo principal é avaliar como diferentes operadores de borda (como Sobel, Roberts, Prewitt e Canny) e suas variações de hiperparâmetros (tamanho de kernel, sigmas e limiares) impactam a precisão do resultado final em comparação com o esforço de processamento exigido.

### **O que está sendo analisado?**

* **Qualidade da Detecção:** Avaliada através de métricas clássicas de visão computacional: Precision, Recall e F1-Score. As predições são validadas contra o Ground Truth (gabarito) do dataset BIPEDv2.

* **Custo Computacional:** Medido através do tempo de execução médio de cada algoritmo, permitindo identificar o ponto de equilíbrio onde um ganho marginal de qualidade deixa de justificar o aumento no tempo de processamento.

### **Base de Dados escolhida**

**Dataset Barcelona Images for Perceptual Edge Detection (BIPED)**

O conjunto de dados [BIPID](https://github.com/xavysp/MBIPED?tab=readme-ov-file) contém 250 imagens externas de 1280 × 720 pixels cada. Essas imagens foram cuidadosamente anotadas por especialistas em visão computacional, portanto, nenhuma redundância foi considerada. Apesar disso, todos os resultados foram verificados diversas vezes para corrigir possíveis erros ou bordas incorretas atribuídas a apenas um dos participantes. Este conjunto de dados é de autoria de Xavier Soria, Edgar Riba e Angel Sappa, e está disponível publicamente como referência para avaliação de algoritmos de detecção de bordas.

 A criação deste conjunto de dados foi motivada pela escassez de conjuntos de dados para detecção de bordas; na verdade, existe apenas um conjunto de dados disponível publicamente para essa tarefa, publicado em 2016 (MDBD: Multicue Dataset for Boundary Detection — o subconjunto para detecção de bordas). O nível de detalhamento das anotações de borda nas imagens do BIPED pode ser apreciado observando o gráfico de dispersão (GT), conforme mostrado nas figuras acima.

O conjunto de dados BIPED possui 250 imagens em alta definição. Essas imagens já estão divididas em conjuntos de treinamento e teste: 200 para treinamento e 50 para teste.

### **Como Executar o Experimento** 


* **1) Clonar o Repositório**
Primeiro, clone o projeto para a sua máquina local:

```bash
git clone https://github.com/Nycksandro/TP_AVD
cd TP_AVD
```

* **2) Configurar o Ambiente Virtual (venv)**
Recomenda-se o uso de um ambiente isolado para evitar conflitos de dependências.

```bash
# Criar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate # Ativar o ambiente No Linux:

#.venv\Scripts\activate # No Windows:
```

* **3) Instalar as dependências necessárias** 
Com o ambiente ativo, instale as bibliotecas necessárias.

```bash 
pip install -r requirements.txt 
```

* **4) Obter o Dataset(BIPEDv2)**
O projeto requer o dataset BIPEDv2 organizado na raiz do projeto.

* **Download via Google Drive:** [Dataset BIPEDv2](https://drive.google.com/file/d/1vvts9m82PpgMz-nRPPI800DuE9svOfsY/view?usp=drive_link)
* **Download alternativo (opcional):** Se preferir via terminal (requer `gdown` instalado):

```bash
pip install gdown 
gdown 1vvts9m82PpgMz-nRPPI800DuE9svOfsY # Link da pasta do Dataset
unzip BIPEDv2.zip # Descompactando
```

**Importante:** Após o download, certifique-se de que as pastas de imagens e gabaritos estão localizadas dentro do diretório `BIPEDv2` na raiz, conforme a estrutura abaixo:

```bash
 ┣ 📂BIPEDv2
 ┃ ┣ 📂BIPED
 ┃ ┃ ┣ 📂edges
 ┃ ┃ ┃ ┣ 📂edges_maps
 ┃ ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┃ ┃ ┣ 📂rgbr
 ┃ ┃ ┃ ┃ ┣ 📂train
 ┃ ┃ ┃ ┃ ┃ ┣ 📂rgbr
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ 📂real 
 ┃ ┃ ┃ ┣ 📂imgs
 ┃ ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┃ ┃ ┣ 📂rgbr
 ┃ ┃ ┃ ┃ ┣ 📂train
 ┃ ┃ ┃ ┃ ┃ ┣ 📂rgbr 
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ 📂real
```
 
* **5) Ambiente final** 
Dessa forma, seu projeto deve estar organizado dessa maneira.

```bash
 ┣ 📂BIPEDv2
 ┃ ┣ 📂BIPED
 ┃ ┃ ┣ 📂edges
 ┣ 📜Experimento.py
 ┣ 📜README.md
 ┣ 📜tarefa01.py
 ┣ 📜requirements.txt
``` 

* **6) Rodar o mini_estudo01.py**

```bash
python3 mini_estudo01.py
```
