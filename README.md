# Matemática Aplicada

Repositório de estudos de matemática aplicada com foco em machine learning, ciência de dados e fundamentos matemáticos para modelagem.

A ideia é organizar revisões, notebooks e experimentos práticos sobre os temas que sustentam algoritmos de ML: álgebra linear, cálculo/otimização, probabilidade, estatística e aplicações em dados.

## Objetivo

Este repositório funciona como um caderno de estudos técnico: cada bloco busca conectar a intuição matemática com implementação em Python, usando exemplos pequenos, visualizações e aplicações em ciência de dados.

O foco é revisar os fundamentos sem deixá-los soltos, mostrando como eles aparecem em problemas reais de modelagem, otimização, regressão e análise de dados — e como os blocos se conectam entre si (ex: mínimos quadrados é, ao mesmo tempo, projeção em álgebra linear, gradiente zero em cálculo, e máxima verossimilhança em probabilidade).

## Estrutura atual

```text
.
├── algebralinear/
│   ├── algebra_linear.ipynb      # transformações lineares, matrizes, determinante, posto,
│   │                             # inversa, transposta, produto interno, ortogonalidade,
│   │                             # projeção, mínimos quadrados, autovalores/autovetores, PCA
│   └── RESUMO.MD                 # guia teórico completo do tema, conectando tudo
├── calculo/
│   ├── calculo_otimizacao_visual.ipynb  # derivada, regra da cadeia, gradiente,
│   │                                     # gradient descent/SGD, Lagrange — versão visual
│   └── RESUMO.MD                 # guia teórico do bloco de cálculo/otimização
├── estatistica/
│   └── RESUMO.MD                 # guia teórico de probabilidade e estatística
│                                  # (Bayes, distribuições, MLE, viés-variância, testes de hipótese)
├── machine_learning/
│   ├── 1.intro/
│   │   ├── numpy.ipynb           # arrays, indexação, slicing, matrizes, números aleatórios
│   │   └── pandas.ipynb          # DataFrame, Series, seleção, indexação (loc/iloc), filtros
│   └── 2.predict-car-price/
│       ├── predict-car-price.ipynb  # projeto de regressão: EDA, split treino/val/teste,
│       │                            # transformação log do target, regressão linear do zero
│       │                            # (loop, produto escalar, forma matricial)
│       └── data.csv              # dataset de preços de carros usado no projeto
├── myprojects/
│   └── README.MD                 # espaço para projetos originais de estudo pessoal
└── README.md
```

## Conteúdo por tema

| Pasta | Tema | O que tem |
| --- | --- | --- |
| `algebralinear/` | Álgebra linear | Notebook + `RESUMO.MD` cobrindo transformação linear, matrizes, determinante, posto, inversa, transposta, produto interno, ortogonalidade, projeção, mínimos quadrados, autovalores/autovetores e PCA. |
| `calculo/` | Cálculo & otimização | Notebook visual + `RESUMO.MD` cobrindo derivada, regra da cadeia (backprop), gradiente, gradient descent/SGD e multiplicadores de Lagrange. |
| `estatistica/` | Probabilidade & estatística | `RESUMO.MD` cobrindo Bayes, distribuições (Bernoulli/Binomial/Normal), esperança/variância/covariância, MLE, viés-variância e testes de hipótese. Ainda sem notebook prático. |
| `machine_learning/1.intro/` | Fundamentos de Python para dados | Notebooks de NumPy e Pandas — a base manipulada nos projetos seguintes. |
| `machine_learning/2.predict-car-price/` | Projeto aplicado | Regressão linear implementada do zero (sem scikit-learn) para prever preço de carros: EDA, split de dados, transformação logarítmica do alvo, e a evolução da fórmula até a forma matricial `ŷ = Xw`. |
| `myprojects/` | Projetos pessoais | Espaço reservado para projetos originais baseados em interesses pessoais. |

## Como executar

1. Clone o repositório:

```bash
git clone <url-do-repositorio>
cd Matematica-aplicada
```

2. Crie um ambiente virtual, se desejar:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instale as dependências principais:

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

4. Abra o notebook desejado, por exemplo:

```bash
jupyter notebook machine_learning/2.predict-car-price/predict-car-price.ipynb
```

Também é possível abrir os notebooks diretamente pelo VS Code com a extensão de Jupyter instalada.

## Dependências

Os notebooks atuais usam principalmente:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

## Linha de estudo

1. Revisar os fundamentos matemáticos de cada bloco (álgebra linear → cálculo → probabilidade).
2. Implementar exemplos simples em Python, sem depender de bibliotecas de ML prontas.
3. Conectar cada conceito com aplicações reais em dados (ex: regressão linear do zero no projeto de preço de carros).
4. Criar visualizações para reforçar a intuição geométrica.
5. Consolidar os temas em notebooks e resumos organizados e reutilizáveis.

## Status

- **Álgebra linear** e **Cálculo/otimização**: base teórica fechada, com `RESUMO.MD` extenso e notebook de apoio.
- **Probabilidade/estatística**: base teórica fechada em `RESUMO.MD`, ainda sem notebook prático.
- **Machine learning aplicado**: em desenvolvimento — projeto de predição de preço de carros implementando regressão linear do zero, conectando com os fundamentos de álgebra linear.

Em desenvolvimento contínuo como material pessoal de revisão e prática em matemática aplicada para machine learning e ciência de dados.
