# AVD - Experimento

* **Detectores com implementações disponíveis**

#### **1) Roberts**

    É o detector de bordas mais minimalista. Ele utiliza kernels 2×2 para calcular o gradiente em uma base diagonal.

    Funcionamento: Calcula a diferença entre pixels diagonalmente adjacentes.

    Trade-off: É o mais rápido de todos devido ao tamanho reduzido do kernel, mas é extremamente sensível ao ruído e produz bordas muito "serrilhadas". 

#### **2) Sobel**

    É o detector de bordas mais minimalista. Ele utiliza kernels 2×2 para calcular o gradiente em uma base diagonal.

    Funcionamento: Calcula a diferença entre pixels diagonalmente adjacentes.

    Trade-off: É o mais rápido de todos devido ao tamanho reduzido do kernel, mas é extremamente sensível ao ruído e produz bordas muito "serrilhadas".
       

#### **3) Canny**

    Não é apenas um filtro, mas um pipeline de processamento completo.

    Funcionamento: Passa por quatro etapas: Suavização (Blur), Cálculo de Gradiente, Supressão de Não-Máximos (afina as bordas) e Limiarização por Histerese (conecta bordas fracas a fortes).

    Trade-off: Oferece a melhor qualidade visual (bordas de 1 pixel e contínuas), porém é o mais caro computacionalmente devido à sua natureza multi-estágio.

#### **4) Laplaciano**

    Baseia-se na segunda derivada da imagem. Geralmente aplicado após um filtro Gaussiano (Laplacian of Gaussian).

    Funcionamento: Procura por "cruzamentos por zero" na intensidade. Diferente do Sobel, ele não indica a direção da borda, apenas a presença dela.

    Trade-off: Muito bom para detectar detalhes finos e contornos fechados, mas pode gerar o "efeito espaguete" (linhas onde não há bordas reais) se o ruído não for bem tratado.

#### **5) Operador de Scharr**

    Uma otimização direta da matemática do Sobel.

    Funcionamento: Utiliza kernels 3×3 com pesos projetados para serem mais "isotrópicos" (terem a mesma resposta independente da inclinação da borda). Enquanto o Sobel usa pesos como 1, 2, 1, o Scharr usa 3, 10, 3.

    Trade-off: Possui exatamente o mesmo custo computacional do Sobel, mas entrega resultados significativamente melhores para bordas diagonais ou curvas.

* **Detectores sem implementações disponíveis**

#### **5) Operador de Frei-Chen**

    Uma abordagem baseada em álgebra linear e projeção de subespaços.

    Funcionamento: Utiliza um conjunto de 9 máscaras 3×3 que representam diferentes estruturas (bordas, linhas, ondulações). O resultado é a projeção da imagem nesse "espaço de bordas".

    Trade-off: Alta precisão em distinguir o que é borda real de o que é apenas ruído ou textura, mas o custo é alto, pois exige a aplicação de 9 convoluções por pixel.

#### **7) Kirsch Compass Kernels**

    Um detector de bordas direcional "compassivo" (direções da bússola).

    Funcionamento: Aplica 8 kernels diferentes, cada um girado em 45°. O valor final do pixel é a maior resposta obtida entre as 8 direções.

    Trade-off: Fornece uma detecção de borda muito robusta e detalhada para qualquer orientação, mas o custo computacional é aproximadamente 4 vezes maior que o do Sobel.  


## **Datasets**

* **Dataset Barcelona Images for Perceptual Edge Detection (BIPED)**: O conjunto de dados BIPED [1] contém 250 imagens externas de 1280 × 720 pixels cada. Essas imagens foram cuidadosamente anotadas por especialistas em visão computacional, portanto, nenhuma redundância foi considerada. Apesar disso, todos os resultados foram verificados diversas vezes para corrigir possíveis erros ou bordas incorretas atribuídas a apenas um dos participantes. Este conjunto de dados está disponível publicamente como referência para avaliação de algoritmos de detecção de bordas. A criação deste conjunto de dados foi motivada pela escassez de conjuntos de dados para detecção de bordas; na verdade, existe apenas um conjunto de dados disponível publicamente para essa tarefa, publicado em 2016 (MDBD: Multicue Dataset for Boundary Detection — o subconjunto para detecção de bordas). O nível de detalhamento das anotações de borda nas imagens do BIPED pode ser apreciado observando o gráfico de dispersão (GT), conforme mostrado nas figuras acima.

O conjunto de dados BIPED possui 250 imagens em alta definição. Essas imagens já estão divididas em conjuntos de treinamento e teste: 200 para treinamento e 50 para teste.

* **Multicue Dataset for Boundary Detection**: Para estudar a interação de várias pistas visuais iniciais (brilho, cor, estéreo, movimento) durante a detecção de limites em cenários naturais desafiadores, usamos os consumidores para construir um conjunto de dados de vídeo multi-clube que consiste em sequências de vídeo binoculares curtas de cenas naturais [2] – câmeras estéreo Classe Fuji (Mély, Kim, McGill, Guo e Serre, 2016). Consideramos uma variedade de locais (de campi universitários a paisagens urbanas e parques) e estações para minimizar possíveis viés. Tentamos capturar cenas mais desafiadoras para detecção de fronteiras, fotografando-as em uma variedade de aparências em cada lente. O quadro-chave de exemplo representativo é mostrado na figura a seguir. O conjunto de dados consiste em 100 cenas, cada uma composta por uma sequência de cores curta (10 quadros) de visualizações esquerda e direita. Cada sequência é amostrada a 30 quadros por segundo. A resolução por quadro é de 1280 x 720 pixels.

## **Fonte**

* [1] [Dataset BIPID](https://github.com/xavysp/MBIPED?tab=readme-ov-file)
* [2] [Dataset MDBD](https://opendatalab.com/OpenDataLab/MDBD)