# Prevendo o tempo de volta na Fórmula 1

Regressão linear implementada do zero para modelar o **ritmo puro** do conjunto piloto-carro no GP da Itália de 2024 (Monza), descontando combustível, pneu e composto.

**Resultado: RMSE de 0,587 s contra um baseline de 1,299 s — 55% de redução.**

Para dar escala: 0,59 s por volta é aproximadamente a diferença entre carros de meio de grid na Fórmula 1.

## O que tem aqui

| Arquivo | O que é |
|---|---|
| `predict-lap-time.ipynb` | **O trabalho.** Análise, derivação da regressão e avaliação |
| `voltas_f1.parquet` | O dataset — 26.604 voltas da temporada 2024 (1 MB) |
| `extrair_voltas.py` | Gera o parquet a partir do cache do FastF1 |
| `baixar_cache_f1.py` | Baixa as sessões da temporada para o cache local |
| `requirements.txt` | `fastf1` e `pyarrow` (Python ≥ 3.10) |

O notebook lê direto o parquet, então **para ler e rodar a análise não é preciso baixar nada**.

## Reproduzir o dataset do zero

```bash
pip install -r requirements.txt
python baixar_cache_f1.py    # baixa as 24 corridas para f1_cache/
python extrair_voltas.py     # gera voltas_f1.parquet
```

⚠️ **Rode isso numa máquina local, não em nuvem.** O servidor de live timing da Fórmula 1 usa CloudFront, que bloqueia faixas de IP de datacenter com erro 403 — Codespaces, Colab e instâncias de nuvem em geral não conseguem baixar. De uma conexão residencial funciona.

O `f1_cache/` fica fora do Git (é pesado e regenerável); o parquet extraído, com poucos MB, é o que vai versionado.

## O achado principal

Os dois coeficientes numéricos contam a história do projeto:

```
lapnumber   -0,065 s/volta    o tanque esvaziando, o carro fica leve
tyrelife    +0,062 s/volta    a degradação do pneu
                     ─────
soma        -0,003 s/volta    ≈ zero
```

Dois efeitos físicos grandes, de sinais opostos e magnitudes quase idênticas. É por isso que a correlação marginal de `tyrelife` com o tempo de volta era $-0{,}034$ — praticamente zero, não porque o pneu não importa, mas porque ele importa **tanto quanto** o combustível, em direção contrária.

Descartar essa feature pela correlação simples custaria 0,062 s/volta invisíveis no modelo — mais de um segundo ao longo de um stint de 20 voltas.

O que permite separar os dois efeitos são as **paradas nos boxes**: dentro de um stint `tyrelife` e `lapnumber` avançam juntos com correlação exatamente 1,0, e só os resets do pit stop descolam as duas colunas (0,56 na corrida inteira).

## Limitações registradas

- **Uma corrida só.** Norris e Piastri saem indistinguíveis do vencedor (margem de ±0,29 s/volta contra diferenças reais de 0,1 a 0,3)
- **Degradação por composto não é respondível** com esses dados: a estratégia decide quando o pneu sai, então o medium só é observado na fase inicial da vida dele (viés de truncamento por seleção)
- **Os intervalos de confiança são otimistas** — supõem erros independentes e variância constante, e ambas as suposições foram medidas e violadas

O próximo passo, descrito no fim do notebook, é expandir para a temporada inteira — o parquet já contém as 24 corridas.

## Dados

[FastF1](https://docs.fastf1.dev/) · temporada 2024 · corridas apenas (não inclui treinos e classificação).
