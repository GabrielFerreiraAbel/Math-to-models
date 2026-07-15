# Probabilidade & Estatística para ML — o meu cérebro

> Terceiro e último bloco da base, construído no mesmo estilo dos outros dois: cada
> conceito só entrou depois de eu **explicar de volta** e a gente afinar. Com as
> minhas palavras nos trechos que eu mesmo construí, e conectado com **álgebra
> linear**, **cálculo** e **ML** — porque os três blocos são um motor só.
>
> Fórmulas em texto (não LaTeX), pra renderizar em qualquer lugar.

---

## A grande imagem (por que este bloco existe)

> **Álgebra linear = o *espaço*. Cálculo = o *movimento*. Probabilidade = a
> *incerteza*.**

Dados reais têm ruído; modelos erram; previsões precisam de honestidade sobre o
que não se sabe. Este bloco é a matemática de **decidir e aprender sob incerteza**.

---

## O fio condutor

> **condicional/Bayes (atualizar crença) → distribuições (as formas da
> aleatoriedade) → esperança/variância/covariância (os resumos) → MLE (achar os
> parâmetros que explicam os dados) → viés-variância (as duas formas de errar) →
> testes de hipótese (efeito real ou acaso?).**

---

## Status

**✅ Fechado:** probabilidade condicional e Bayes · distribuições (Bernoulli,
binomial, normal) · esperança, variância, covariância · máxima verossimilhança
(MLE) · trade-off viés-variância · testes de hipótese e p-valor.

**🟡 Pra fechar quando aparecer:** amostragem na prática (cross-validation,
bootstrap) · correções de multiple testing · conformal prediction (no projeto).

---

# 1. Probabilidade condicional e Bayes

## Condicional — "nichar" o universo

`P(A|B)` = a chance de A **dado que** já sei B.

**Minha intuição:** *B **nicha** o resultado de A — o universo encolhe pra dentro de
B, e ali dentro a probabilidade de A pode ser completamente outra.*
- `P(jogador de basquete)` é minúscula; `P(jogador | tem mais de 2 m)` é bem maior —
  nichou pra um grupo onde jogador é comum.
- `P(jogador | tem nanismo)` ≈ 0 — nichou pro fracasso quase certo.
  *Mesma pessoa, probabilidades diferentes, dependendo do que se condiciona.*

Termo formal do meu "nichar": **restringir o espaço amostral**.
Fórmula: `P(A|B) = P(A e B) / P(B)` — "das vezes em que B acontece (denominador), em
quantas A também acontece (numerador)?"

## Bayes — a máquina de atualizar crença

Quero `P(A|B)`, mas o que consigo medir é `P(B|A)`. Bayes **inverte**:
```
P(A|B) = P(B|A) · P(A) / P(B)
```
- **P(A)** = **prior** — crença antes da evidência
- **P(B|A)** = **verossimilhança** — se A fosse verdade, quão provável seria ver B?
- **P(A|B)** = **posterior** — crença depois da evidência
- **P(B)** = normalização

> **Bayes é a matemática de mudar de ideia com dados:** prior × evidência → posterior.

## A armadilha da doença rara (a lição central)

Doença em 1/1000, teste 99% preciso, deu positivo → chance real ≈ **9%**, não 99%.

**Por quê (a conta que fecha):** em 1000 pessoas → 1 doente (testa positivo: 1
verdadeiro) + 999 saudáveis, com 1% de erro → ~10 **falsos positivos**. Dão positivo
11, só 1 tem a doença → 1/11 ≈ 9%.

**Minha leitura:** *o teste não consegue engolir o prior baixo — ele **complementa**
o prior.* Um prior muito baixo exige evidência **muito** forte pra subir; 99% não
basta contra uma raridade de 1 em 1000. **Posterior = prior × evidência.**

**🔗 ML:** classificadores perguntam `P(classe | dados)` — é Bayes. Naive Bayes é
isso direto (spam). E prior+evidência→posterior é a espinha da inferência bayesiana
e da quantificação de incerteza.

---

# 2. Distribuições — as formas da aleatoriedade

Distribuição = o **mapa** de quais valores saem e com que probabilidade.

## Bernoulli — o tijolo
Um único evento **sim/não** (sucesso=1 com chance `p`, fracasso=0 com `1−p`).
*Um tijolo sozinho não faz casa — é a unidade fundamental de incerteza.*
**ML:** toda classificação binária vive aqui ("70% de chance de ser spam").

## Binomial — a parede
Empilha **n** tijolos de Bernoulli **idênticos e independentes** e **conta** os
sucessos. Parâmetros: `n` e `p`.
> *A parede não quer saber o formato de cada tijolo; quer o resultado do conjunto.*
Formalmente: `Y = X₁ + X₂ + ... + Xₙ` — a binomial é **literalmente a soma** das
Bernoullis. (As condições **i.i.d.** são o que permitem isso.)
Forma: morro com pico perto de `n·p`; extremos raros.

## Normal — a rainha (curva de sino)
**Contínua** (altura, erro, ruído, retorno). Dois parâmetros:
- **μ (média)** = **controle de posição** — *o "GPS" que desliza o sino pra esquerda
  ou direita sem alterar em nada o formato.*
- **σ (desvio-padrão)** = **controle de forma** — *o botão de escala: dita largura e
  altura.*

**σ é a incerteza em pessoa:** σ pequeno → sino estreito e alto → valores concentrados
→ **previsão confiante**. σ grande → sino largo e baixo → valores espalhados →
**muita incerteza**. (Largura e altura se mexem em sentidos opostos porque a área
embaixo é sempre 1 — espalhar a mesma probabilidade obriga a altura a cair.)
Regra prática: ~68% a 1σ, ~95% a 2σ, ~99,7% a 3σ.

## A corrente que liga as três
> **Bernoulli** (1 tentativa) → **Binomial** (soma de n Bernoullis) → **Normal**
> (quando n cresce, a binomial vira o sino).

**Teorema Central do Limite:** somar muita aleatoriedade independente **tende ao
sino**, não importa a distribuição original. Por isso a normal está em todo lugar.

**🔗 ML:** normal modela ruído e erro (a regressão assume erro normal!), init de
pesos, e é a base da quantificação de incerteza (o `± σ` das previsões).

---

# 3. Esperança, variância e covariância — os resumos

## Esperança — o centro de gravidade
`E[X]` = média de longo prazo = **média ponderada** (`Σ valor × probabilidade`).
É o ponto onde a distribuição **se equilibra**. Dado justo: `E[X] = 3.5` (valor que
nunca sai, mas é o centro).

## Variância — o espalhamento
`Var(X) = E[(X − E[X])²]` — a **média das distâncias ao quadrado até o centro**.
É `σ²`; o desvio-padrão é a raiz.

**Minha sacada:** *é rigorosamente o mesmo truque dos Mínimos Quadrados.* Lá media-se
o erro dos pontos até a **reta**; aqui, o espalhamento dos dados até a **média**. O
quadrado existe pra os desvios não se anularem (e pra penalizar mais os grandes).

## Covariância — a dança entre duas variáveis
`Cov(X,Y) = E[(X − E[X])(Y − E[Y])]` — a média do **produto dos desvios**.
- positiva: sobem juntas · negativa: sentidos opostos · zero: sem relação linear

**Minha sacada (a conexão de ouro):** *a covariância é o **produto interno** dos
vetores de dados **depois de mover a origem para a média** de cada variável. O
produto interno mede o alinhamento de **setas**; a covariância mede o alinhamento
das **flutuações**.* Covariância zero → os vetores de flutuação são **ortogonais**.
> **Descorrelação é a ortogonalidade da probabilidade.**

**⚠️ Ajuste que travei (importante):** covariância zero = **descorrelacionadas**,
**não** necessariamente **independentes**. Independência é mais forte. Contraexemplo:
`Y = X²` com X simétrico → Y é *totalmente* determinado por X (dependência máxima),
mas `Cov(X,Y) = 0`, porque a covariância só enxerga relação **linear**.
> **Independentes ⟹ descorrelacionadas. Descorrelacionadas ⇏ independentes.**
(É por isso que métodos lineares perdem padrões curvos que métodos não-lineares pegam.)

**🔗 Álgebra linear:** a **matriz de covariância** é a tabela de "quem dança com
quem" — simétrica e PSD (`xᵀXᵀXx = ‖Xx‖² ≥ 0`), e seus **autovetores** são as
componentes principais → **PCA**. Covariância → simétrica → autovetores ortogonais →
PCA. Tudo se junta.

---

# 4. Máxima Verossimilhança (MLE)

## A inversão
Probabilidade pergunta: "dados os parâmetros, qual a chance destes dados?"
MLE inverte: **"dado que observei ESTES dados, quais parâmetros os tornam mais
prováveis?"** (mesmo espírito da inversão do Bayes.)

**Minhas palavras:** *procura o θ que maximiza a probabilidade de termos observado
exatamente o dataset que temos em mãos. É a resposta para: "de todos os mundos
possíveis, qual deles geraria estes dados com maior facilidade?"* — o **detetive**
escolhendo a hipótese que torna as evidências menos surpreendentes.

## O mecanismo
Se os dados são **i.i.d.**, a probabilidade conjunta é o **produto**:
```
L(θ) = P(dado₁|θ) · P(dado₂|θ) · ... · P(dadoₙ|θ)
```
**O truque do log** (três razões, todas minhas):
1. **Transforma produtos em somas** (`log(A·B) = log A + log B`) → derivar fica fácil.
2. **Salva a precisão numérica** — multiplicar milhares de frações vira soma de
   números manejáveis.
3. **Não altera o ponto máximo** — o log é estritamente crescente, então o θ que
   maximiza `log L` é o mesmo que maximiza `L`.

Vira: **maximizar `Σ log P(dadoᵢ|θ)`** → derivar e igualar a zero: `∇θ log L(θ) = 0`.

**⚠️ Ajuste que travei:** os parâmetros **só "saltam da fórmula" quando há solução
fechada** (regressão linear com erro normal → cai o `XᵀX ŵ = Xᵀb`). Na maioria dos
modelos (logística, redes neurais) **não há fórmula fechada** — aí `∇log L = 0` não
se resolve na mão e é preciso **subir/descer a rampa** com gradient descent
(maximizar L = minimizar `−log L`, a "loss"/cross-entropy).

## 🔗 O arco que fecha os três blocos
> **Se o erro é normal, MLE = minimizar o erro ao quadrado = mínimos quadrados.**

A regressão é:
- **projeção** no espaço-coluna (álgebra linear) — `Xᵀe = 0`
- **descer a rampa de erro** (cálculo) — `∇erro = 0`
- **máxima verossimilhança com erro normal** (probabilidade) — `∇log L = 0`

**Três rostos do mesmo problema.** E treinar qualquer modelo em ML é maximizar uma
verossimilhança (ou minimizar seu negativo) — a "loss" quase sempre **vem** de uma MLE.

---

# 5. Trade-off viés-variância — as duas formas de errar

## Os dois estudantes ruins (minha analogia)

**🧠 O Cabeça-Dura (alto viés / underfitting):** *preguiçoso e simplista, adota uma
regra rígida pra tudo — "foi por causa da guerra" pra qualquer questão. Erra tanto
no simulado (treino) quanto na prova (teste); a nota é consistentemente baixa.*
Na IA: assume que os dados são simples demais (reta pra dados que formam curva).

**📝 O Decoreba (alta variância / overfitting):** *decora cada linha das provas
antigas — inclusive os erros de digitação e as **manchas de café**. Tira 10 no
simulado idêntico, mas se mudar uma vírgula, entra em pânico e tira zero.*
Na IA: complexo demais, **memorizou o ruído** em vez do padrão real.

> As "manchas de café" são exatamente o que **ruído** significa: acidentes daquele
> dataset específico, que não são o padrão.

## Por que é trade-off
O botão é a **complexidade do modelo**:
- **Mais complexo:** viés **cai** (aprende nuances) · variância **sobe** (memoriza ruído)
- **Menos complexo:** variância **cai** (fica estável) · viés **sobe** (volta a ser cabeça-dura)

Não dá pra zerar os dois: **abaixar um levanta o outro.**

## O que completa o quadro
- **O fundo do U:** o erro total é um **U** — o alvo é a **complexidade ótima** no
  meio, onde a soma é mínima (não minimizar um dos dois isolado).
- **Erro total = viés² + variância + ruído irredutível.** O terceiro é a
  aleatoriedade genuína do mundo — **nenhum** modelo elimina. Serve de humildade:
  impede perseguir erro zero e ensina quando o modelo já está bom o suficiente.

**🔗 Prática:** erro alto no treino **e** no teste = viés (underfitting) → mais
complexidade. Erro baixo no treino e alto no teste = variância (overfitting) →
**regularização** (ridge/lasso segura a variância) ou mais dados.

---

# 6. Testes de hipótese e p-valor

## A lógica do advogado do diabo
- **H₀ (nula)** = *a presunção de inocência: o réu é inocente, o remédio não funciona,
  a moeda é justa. O mundo continua sem novidades.*
- **Hₐ (alternativa)** = *a acusação: há algo de novo.*

*A estatística é extremamente cética: assume H₀ rigorosamente verdadeira, coleta os
dados, e se eles forem absurdamente improváveis sob essa hipótese, é obrigada a
rejeitar H₀.* — uma **prova por contradição probabilística**.

## O p-valor
> **p-valor = P(dados tão extremos quanto os meus | H₀ é verdadeira).**

*p pequeno (0.01): se H₀ fosse verdadeira, a chance de ver dados tão bizarros seria
de 1%. Como é quase impossível sob a "mentira" de H₀, derrubamos H₀.*
(O 0.05 é convenção, não lei da natureza.)

## ⚠️ A armadilha (o erro estatístico mais comum do mundo)
**p-valor ≠ "probabilidade de ter sido acaso".** É a mesma inversão do Bayes:
```
p-valor        = P(Dados | H₀)          ✔ o que é
interpretação  = P(H₀ | Dados)          ✘ o que todos querem que fosse
```

**O réu no cofre (minha ilustração):**
- H₀: o suspeito é inocente. D: a digital dele está dentro do cofre.
- `P(D|H₀)`: se inocente, qual a chance da digital estar lá? **Minúscula** → p-valor pequeno.
- `P(H₀|D)`: dado a digital, qual a chance de ser inocente? **Pode ser altíssima** —
  *se ele for o técnico que instalou o cofre semana passada!*

Pra converter um no outro é **obrigatório** usar Bayes — e Bayes exige o **prior**
`P(H₀)`, que o p-valor **não tem**:
```
P(H₀|Dados) = P(Dados|H₀) · P(H₀) / P(Dados)
```
> É o **mesmo** erro da doença rara: sem o prior, você não sabe o posterior.

## Os dois jeitos de errar (o eco do viés-variância)
- **Falso positivo (Tipo I):** rejeitar H₀ verdadeira — "descobrir" efeito inexistente.
- **Falso negativo (Tipo II):** não rejeitar H₀ falsa — perder uma descoberta real.

**🔗 ML:** comparar modelos (A é melhor que B, ou foi sorte do split?) · **multiple
testing** (testar 100 hipóteses a 5% → ~5 falsos positivos por acaso) · e a base do
raciocínio da **conformal prediction** (o que é normal vs. surpreendente) — a
incerteza honesta do **river-lifelines**.

---

## Mapa de interligações

**Dentro do bloco:**
condicional → Bayes (atualizar crença) → distribuições (as formas) →
esperança/variância/covariância (os resumos) → MLE (achar θ) → viés-variância
(as 2 formas de errar) → testes de hipótese (real ou acaso?) → *volta ao Bayes*
(o p-valor mal-lido é a inversão do Bayes).

**Probabilidade ↔ Álgebra Linear:**
| Probabilidade | é / liga com | Álgebra Linear |
|---|---|---|
| covariância | é | **produto interno** dos desvios (dados centralizados) |
| descorrelação (Cov=0) | é | **ortogonalidade** |
| matriz de covariância | é | **simétrica e PSD** → autovetores → **PCA** |
| variância (quadrado) | mesmo truque de | mínimos quadrados (‖e‖²) |

**Probabilidade ↔ Cálculo:**
| Probabilidade | vira | Cálculo |
|---|---|---|
| maximizar verossimilhança | é | otimização: `∇log L = 0` |
| sem fórmula fechada | exige | **gradient descent** (na loss = −log L) |
| log (produto→soma) | facilita | derivar termo a termo |

**Probabilidade ↔ ML:**
Bayes = classificadores · normal = modelo do erro/ruído · MLE = **de onde vêm as
loss functions** · viés-variância = under/overfitting e regularização ·
testes = comparar modelos, conformal prediction.

---

## Mini-checks (Feynman)

1. Por que `P(A|B)` difere de `P(A)`? → B **nicha**/restringe o espaço amostral.
2. Teste 99% + doença rara = 9%. Por quê? → prior baixíssimo; falsos positivos
   afogam o verdadeiro (10 contra 1). Posterior = prior × evidência.
3. Binomial e Bernoulli? → binomial é a **soma de n Bernoullis i.i.d.** (tijolo→parede).
4. O que σ diz? → a **incerteza**: pequeno = confiante; grande = espalhado.
5. Por que o quadrado na variância? → mesmo truque dos mínimos quadrados (desvios
   não se anulam).
6. Covariância e produto interno? → covariância é o produto interno dos **desvios**;
   zero = flutuações **ortogonais** (mas ≠ independentes!).
7. O que a MLE procura? → o mundo (θ) que geraria estes dados com maior facilidade.
8. Por que log na MLE? → produto→soma, precisão numérica, não muda o argmax.
9. MLE = mínimos quadrados quando? → quando o **erro é normal**.
10. Viés vs variância? → cabeça-dura (simples demais) vs decoreba (memoriza o ruído).
11. p-valor é a chance de ser acaso? → **NÃO**: é `P(dados|H₀)`, não `P(H₀|dados)`
    (o réu no cofre / o técnico).

---

## Próximos passos

1. **Curso do IMPA** — a base está completa; agora cada conceito novo pousa em algo
   que já entendo.
2. **river-lifelines (Passo 3)** — aplicar: a regressão (projeção + MLE), features,
   regime-shift (diagnosticado com viés-variância), incerteza (conformal).
3. Fechar quando aparecer: cross-validation/bootstrap na prática, multiple testing,
   conformal prediction.

---

> **Fechamento — a base está completa.** Álgebra linear = o espaço; cálculo = o
> movimento; probabilidade = a incerteza. E os três se encontram no mesmo ponto: a
> regressão é **projeção**, é **descer a rampa**, e é **máxima verossimilhança**.
> Não estudei três matérias — estudei uma coisa só, de três ângulos.