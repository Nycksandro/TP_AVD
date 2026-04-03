## **Experimento de AVD**

Este repositório contém um estudo experimental focado na análise do trade-off entre qualidade de detecção e custo computacional em algoritmos de processamento digital de imagens.

O objetivo principal é avaliar como diferentes operadores de borda (como Sobel, Roberts, Prewitt e Canny) e suas variações de hiperparâmetros (tamanho de kernel, sigmas e limiares) impactam a precisão do resultado final em comparação com o esforço de processamento exigido.

### **O que está sendo analisado?**

* **Qualidade da Detecção:** Avaliada através de métricas clássicas de visão computacional: Precision, Recall e F1-Score. As predições são validadas contra o Ground Truth (gabarito) do dataset BIPEDv2.

* **Custo Computacional:** Medido através do tempo de execução médio de cada algoritmo, permitindo identificar o ponto de equilíbrio onde um ganho marginal de qualidade deixa de justificar o aumento no tempo de processamento.

### **Como Executar o Experimento** 

Para executar o experimento, é necessário executar as seguintes etapas:

* **1) Organizar o ambiente:** A sua pasta deve estar assim

```bash
 ┣ 📂BIPEDv2
 ┃ ┣ 📂imagens
 ┃ ┣ 📂gabarito
 ┣ 📜Experimento.py
 ┣ 📜Experimento.py
 ┣ 📜README.md
 ┣ 📜avd_experimento.md
 ┣ 📜main.py
 ┣ 📜requirements.txt
``` 
E para isso, baixe o dataset a seguir e extraia na sua pasta: [Dataset BIPID](https://github.com/xavysp/MBIPED?tab=readme-ov-file)


* **2) Instalar as dependências necessárias** 
```bash 
pip install -r requirements.txt 
```

* **3) Rodar o main.py**
```bash
python3 main.py
```

