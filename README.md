# Matemática Aplicada

Caderno de estudos de matemática aplicada a machine learning, construindo tudo do zero antes de usar biblioteca pronta.

A ideia é não deixar os fundamentos soltos: derivar a matemática, implementar na mão, e só depois comparar com a implementação de referência. Um mesmo resultado aparece de três ângulos — mínimos quadrados é projeção em álgebra linear, gradiente zero em cálculo, e máxima verossimilhança em probabilidade.

## Por onde começar

**Se você tem 5 minutos:** [predict-lap-time.ipynb](myprojects/1.f1-lap-time-prediction/predict-lap-time.ipynb) — prever tempo de volta na Fórmula 1 com regressão derivada do zero. RMSE de 0,587 s contra baseline de 1,299 s. É o trabalho mais completo aqui.

**Se quer ver a base teórica:** os três `RESUMO.MD` de [álgebra linear](algebralinear/RESUMO.MD), [cálculo](calculo/RESUMO.MD) e [estatística](estatistica/RESUMO.MD). Cada conceito foi escrito depois de eu conseguir explicá-lo de volta, com as conexões entre os blocos explícitas.

## Estrutura

```text
.
├── algebralinear/       transformação linear, determinante, posto, inversa,
│                        produto interno, ortogonalidade, projeção,
│                        mínimos quadrados, autovalores/autovetores, PCA
├── calculo/             derivada, regra da cadeia (backprop), gradiente,
│                        gradient descent/SGD, multiplicadores de Lagrange
├── estatistica/         Bayes, distribuições, esperança/variância/covariância,
│                        MLE, viés-variância, testes de hipótese
├── machine_learning/    trilha aplicada, seguindo material de curso
│   ├── 1.intro/            NumPy e Pandas
│   └── 2.predict-car-price/  regressão do zero, ridge, equação normal
└── myprojects/          projetos originais, problema escolhido por mim
    └── 1.f1-lap-time-prediction/
```

Cada pasta de fundamento tem um notebook prático e um `RESUMO.MD` com a teoria conectada.

## Notebooks

| Notebook | Tema |
|---|---|
| [algebra_linear.ipynb](algebralinear/algebra_linear.ipynb) | transformações, projeção, mínimos quadrados, autovetores, PCA |
| [calculo_otimizacao_visual.ipynb](calculo/calculo_otimizacao_visual.ipynb) | derivada, gradiente, gradient descent, Lagrange — versão visual |
| [numpy.ipynb](machine_learning/1.intro/numpy.ipynb) | arrays, indexação, slicing, matrizes, aleatórios |
| [pandas.ipynb](machine_learning/1.intro/pandas.ipynb) | DataFrame, Series, seleção, `loc`/`iloc`, filtros |
| [predict-car-price.ipynb](machine_learning/2.predict-car-price/predict-car-price.ipynb) | regressão do zero, equação normal, regularização |
| [predict-lap-time.ipynb](myprojects/1.f1-lap-time-prediction/predict-lap-time.ipynb) | projeto original de F1 — modelagem e investigação |

## Os dois projetos

**[predict-car-price.ipynb](machine_learning/2.predict-car-price/predict-car-price.ipynb)** — regressão linear implementada do zero. Transformação logarítmica do alvo (cauda longa), evolução da fórmula do loop até a forma matricial $\hat{y} = Xw$, e a descoberta da armadilha da variável dummy: o RMSE explodiu para 498 por colinearidade perfeita, diagnosticada pelo número de condição da matriz e resolvida com regularização.

**[predict-lap-time.ipynb](myprojects/1.f1-lap-time-prediction/predict-lap-time.ipynb)** — problema escolhido por mim, dados via FastF1. Além do modelo, o notebook investiga *por que* os números saem como saem: um confundimento entre combustível e degradação de pneu que zera a correlação marginal, um viés de truncamento por estratégia de corrida, e a medição de quanto o corte aleatório de validação infla o resultado. Resultados negativos e hipóteses que falharam estão documentados.

## Como executar

```bash
git clone <url-do-repositorio>
cd Matematica-aplicada

python3 -m venv .venv
source .venv/bin/activate

pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

O projeto de F1 tem dependências próprias (`fastf1`, `pyarrow`) listadas no [requirements.txt](myprojects/1.f1-lap-time-prediction/requirements.txt) dele.

Os notebooks rodam de cima a baixo sem precisar baixar nada — os dados já estão versionados.

## Linha de estudo

1. Revisar os fundamentos matemáticos (álgebra linear → cálculo → probabilidade)
2. Implementar do zero, sem biblioteca de ML pronta
3. Aplicar em problemas reais e escolhidos por mim
4. Comparar com a implementação de referência, para validar o que foi construído

## Status

| Bloco | Situação |
|---|---|
| Álgebra linear | base fechada — [RESUMO.MD](algebralinear/RESUMO.MD) e [notebook](algebralinear/algebra_linear.ipynb) |
| Cálculo & otimização | base fechada — [RESUMO.MD](calculo/RESUMO.MD) e [notebook visual](calculo/calculo_otimizacao_visual.ipynb) |
| Probabilidade & estatística | base fechada em [RESUMO.MD](estatistica/RESUMO.MD), sem notebook prático ainda |
| ML aplicado | preço de carros concluído |
| Projetos originais | F1 (Monza 2024) concluído; próximo passo é expandir para a temporada inteira |

Em desenvolvimento contínuo.
