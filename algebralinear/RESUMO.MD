# Álgebra Linear para ML — o meu cérebro (completo)

> Meu mapa de estudo, construído bloco por bloco, com as minhas palavras e
> analogias. Cada conceito só entrou aqui depois de eu **explicar de volta** e a
> gente afinar até virar entendimento — não decoreba.
>
> Fórmulas em texto/código (não LaTeX), pra renderizar em qualquer lugar
> (Obsidian, GitHub, VSCode). E o documento é **honesto sobre o status**.

---

## Onde estou (status)

**✅ Sólido:** transformação linear · multiplicação como composição · determinante ·
posto/espaço-coluna · inversa · transposta · produto interno · ortogonalidade ·
**projeção / regressão como projeção** (com a ponte cálculo↔geometria) ·
(bônus) produto vetorial · **matrizes simétricas e positivas definidas** · **SVD**.

**🟡 Falta drillar:** a **mecânica dos autovalores** (`det(A − λI) = 0`). Tenho a
intuição (a "pista") e usei autovetores em tudo (simétricas, PSD, SVD), mas não
derivei essa fórmula a fundo.

**⏳ Não comecei:** cálculo/otimização (gradiente, Jacobiano, Hessiana, gradient
descent) · probabilidade/estatística (Bayes, distribuições, MLE, viés-variância).

---

## O fio condutor (a história em uma linha)

> **Matrizes transformam o espaço → o produto interno mede alinhamento → a
> ortogonalidade diz quando não há sobreposição → a projeção acha a melhor
> aproximação → os autovetores revelam as direções que a transformação respeita →
> a SVD generaliza isso pra qualquer matriz.**

Cada bloco é o anterior visto de outro ângulo.

---

# 1. Transformação linear ✅

Uma matriz não é tabela de números — é uma **transformação do espaço** (estica,
gira, cisalha). O segredo: as **colunas são o destino da base**. 1ª coluna = para
onde î vai; 2ª = para onde ĵ vai. Qualquer vetor pega carona: `v = 2î+1ĵ` vira
`2·(destino de î) + 1·(destino de ĵ)`.

Linear = respeita `T(u+v)=T(u)+T(v)` **e** `T(c·v)=c·T(v)`. As duas.

**Ajuste que travei:** linearidade obriga a **origem a ficar fixa** (`T(0)=0`). Por
isso `Wx` é linear e `Wx+b` **não** é — é *afim*; o bias `b` desloca a origem.
Teste de 1s: `f(0)≠0` → não é linear.

**ML:** uma camada de rede é `W @ x`.

---

# 2. Multiplicação = composição ✅

`A@B` = "aplica **B** primeiro, depois **A**", lido da **direita pra esquerda**
(igual `f(g(x))`). Mecânica: linha·coluna. Mas o que importa é compor
transformações em sequência.

**Ajuste:** por isso **não comuta** (`AB ≠ BA`) — girar-depois-esticar ≠
esticar-depois-girar. (Em rede profunda `W₃W₂W₁x`, trocar camadas muda a função.)

---

# 3. O triângulo: determinante ⟺ posto ⟺ inversa ✅

**Determinante:** mede quanto a transformação **escala área/volume**; sinal =
orientação; **`det=0` = colapso** (o plano vira reta, informação perdida).

**Posto:** a **dimensão do espaço-coluna** — quantas direções sobrevivem.
> Ajuste: posto é a dimensão que a saída *realmente ocupa*, pode ser **menor** que
> o ambiente (3×3 que colapsa num plano tem posto 2).

**Espaço-coluna:** o *span* das colunas = todas as saídas possíveis (`Ax`) = a imagem.

**Inversa:** **desfaz** a transformação; `x = A⁻¹v`; `A⁻¹A = I`.
> Ajuste: é `x = A⁻¹v` (a inversa multiplica o **v**), e só existe se `det≠0` — se
> colapsou, infinitos pontos caem no mesmo lugar, não dá pra saber de onde vieram.

**O triângulo de ferro:**
```
posto cheio  ⟺  det ≠ 0  ⟺  A invertível  ⟺  Ax = b tem solução única
```
Se uma cai, todas caem.

**ML:** `np.linalg.matrix_rank`, `np.linalg.det`, `np.linalg.inv`.

---

# 4. Transposta ✅

`Aᵀ` **espelha** pela diagonal (entrada (i,j) vira (j,i)). **Simétrica** = `A=Aᵀ`.

**Por que `(AB)ᵀ = BᵀAᵀ` (o elevador, minha analogia):** pra voltar de Z pra X, Bᵀ
só fala a língua de Y — estou preso em Z. Primeiro Z→Y com Aᵀ, depois Y→X com Bᵀ.
A ordem inverte porque o caminho de volta desfaz **de fora pra dentro**.

**Ajuste (chave):** transposta **NÃO desfaz** (isso é a inversa) — ela só
**espelha**. E transpor age em **um objeto só** (deitar o vetor), **não** é
multiplicar por nada.

**Dualidade:** transpor um vetor-coluna o vira numa transformação-linha (1×n) que
come um vetor e cospe um número — o `xᵀ` é o x virando "máquina de medir
alinhamento". Ponte pro produto interno.

**ML:** aparece em `XᵀX`, `xᵀy`, e no backprop (transpostas no backward pass).

---

# 5. Produto interno ✅

`aᵀb` mede a **cumplicidade direcional** — um **termômetro de alinhamento**. Entra
dois vetores, sai um número: `aᵀb = ‖a‖‖b‖cos(θ)`.
- alinhados → positivo grande · perpendiculares → zero · opostos → negativo.

O `ᵀ` é o que "deita" o a pra `(1×n)·(n×1)` dar um escalar. **ML:** `a @ b`.

---

# 6. Ortogonalidade ✅

`aᵀb = 0` → perpendiculares. Leitura profunda (minha): **saber tudo sobre `a` dá
ZERO pistas sobre a direção de `b`** — informação não-redundante.

**Ajuste (a virada):** não é "direção diferente" — é **sobreposição zero**. Dois
vetores a 30° também têm direção diferente, mas um ainda tem componente no outro.
Ortogonal = **nenhuma** componente. Features redundantes = produto interno **alto**
(muita sobreposição = mesma informação repetida).

> O `aha`: **sobreposição = produto escalar.** O produto interno mede *quanto* se
> sobrepõem; zero é o único ponto sem sobreposição nenhuma.

**ML:** PCA usa direções ortogonais pra não repetir variância; base ortonormal
(`QᵀQ=I`) torna projeção trivial; rotação é `Qᵀ = Q⁻¹`.

---

# 7. Projeção e regressão — o grande prêmio ✅

**O átomo:** a projeção de `b` na direção de `a` é a **sombra** de b na reta —
o ponto mais próximo. É um múltiplo `x̂a`, e o que sobra é o **erro** `e = b − x̂a`.

**A condição central:** o erro é **perpendicular** (`aᵀe = 0`). Por quê? Se batesse
torto, sobraria um pedaço "deitado" na reta e daria pra escorregar e chegar mais
perto. Só o perpendicular não tem pra onde escorregar → **mais próximo ⟺ erro
perpendicular.** Disso cai `x̂ = aᵀb/aᵀa` e a **decomposição ortogonal**
`b = x̂a + e` (sombra + erro perpendicular).

**O salto pra ML — regressão é projeção:** projeto `b` no **espaço-coluna de X** (o
"cardápio" de previsões `Xŵ`). No exemplo das casas, `b` (preços reais) está
**fora** do cardápio (dado real não cai exato), então acho a sombra. O erro tem que
ser ⊥ a **todas** as colunas — e a transposta empilha isso: cada linha de `Xᵀ` é
uma coluna fazendo produto interno com o erro.
```
Xᵀe = 0  →  Xᵀ(b − Xŵ) = 0  →  XᵀX ŵ = Xᵀb        (equação normal)
```
Não é fórmula mágica — é "erro ⊥ ao espaço das features" em matriz. Minimizar
‖e‖² = achar a sombra = projeção ortogonal.

**A ponte cálculo ↔ geometria (a sacada que fecha tudo):** dá pra achar a mesma
resposta por dois caminhos:
- **Cálculo:** minimizar `∑(yᵢ − axᵢ − b)²` derivando e zerando → `a = cov(x,y)/var(x)`
  (o denominador é o espalhamento de x; o numerador, como x e y variam juntos), e
  `b` sobe/desce a reta pra ela passar no ponto médio `(x̄, ȳ)` — o "centro de
  gravidade".
- **Geometria:** o custo é `‖e‖²`, minimizado quando o erro é ⊥ ao espaço-coluna
  (`Xᵀe = 0` → `XᵀX ŵ = Xᵀb`).

Os dois dão o **mesmo** `a` e `b`, e não é coincidência: *é a mesma física do
espaço operando por trás.* O ponto onde a **rampa de erro fica plana** (derivada
zero) é exatamente o ponto onde a **sombra cai perpendicular** — porque a distância
mínima de um ponto a um plano é sempre a perpendicular. Mecanicamente: **a derivada
de ‖e‖² é, a menos de constante, o `Xᵀe`** — então "gradiente zero" e "erro
ortogonal" são a *mesma equação*. (E regressão simples é só o caso de uma feature
da regressão múltipla — mesma projeção, mais dimensões.)

**Onde tudo reaparece:** resolver exige inverter `XᵀX`, invertível só se as colunas
forem independentes (posto cheio, det≠0). Features redundantes → `XᵀX`
quase-singular → solução explode = **multicolinearidade**.

**ML:**
```python
w = np.linalg.solve(X.T @ X, X.T @ b)
np.allclose(X.T @ (b - X @ w), 0)   # True: erro ⊥ às colunas (a prova)
# np.linalg.lstsq / LinearRegression fazem isso por baixo
```

---

# 8. (bônus) Produto vetorial ✅

Em 3D, `a × b` dá um vetor **perpendicular aos dois**, tamanho = área do
paralelogramo (`‖a‖‖b‖sin θ`). Em 2D vira só um número (área com sinal).
A dimensão vem do nº de coordenadas dos vetores. **Enriquecimento**, não núcleo.
Invariante pra validar: `(a×b)·a = 0` e `(a×b)·b = 0`.

---

# 9. Autovalores e autovetores 🟡 (intuição sólida)

**A "pista":** um autovetor **continua na mesma direção** depois da transformação —
pode esticar, encolher ou inverter, mas não sai da pista. O **autovalor λ** é o
fator dessa escala. Em símbolo: `A v = λ v`.

**Ajustes:** é "não sai da **direção**", não da posição (a ponta se move). Estica a
**própria pista** (qualquer direção, não "x ou y"). Sinais de λ: `>1` estica ·
`0<λ<1` encolhe · `1` mantém · `<0` inverte · `=0` **colapsa** (e aí `det=0`).

**🟡 A mecânica que falta drillar:** de onde saem os λ → `det(A − λI) = 0`. A ideia
(que eu já sinto): procurar λ onde `A − λI` **colapsa** o espaço — colapso é det
zero. Falta eu derivar isso com calma (o polinômio característico) e a
diagonalização `A = QΛQᵀ`.

**ML:** `np.linalg.eig` (`Av=λv`) · `np.linalg.eigh` (simétricas).

---

# 10. Matrizes simétricas e positivas (semi)definidas ✅

**Simétrica (`A = Aᵀ`) — a propriedade mágica:** autovetores **ortogonais entre si**
e autovalores **reais**. Isso é o que faz o PCA funcionar: as componentes principais
são ortogonais *porque* a covariância é simétrica. Cadeia: **simétrica → autovetores
ortogonais → eixos limpos → PCA.**

**Positiva definida — a tigela:** olho `xᵀAx` (pega um vetor, passa pela matriz,
cospe um número) como uma **superfície**. É positiva definida quando `xᵀAx > 0`
para todo `x≠0` → a superfície é uma **tigela** (curva pra cima em toda direção,
mínimo único). Semidefinida = `≥ 0` (pode ter uma calha plana).

**O sinal dos autovalores = a forma** (a sacada):
- todos λ > 0 → **tigela** → mínimo (positiva definida)
- todos λ < 0 → **domo** → máximo (negativa definida)
- **sinais mistos** → **sela** (nem min nem max) — indefinida
- algum λ = 0 → **calha** (semidefinida)
> Ajuste: sela é λ de **sinais mistos**, não "todos negativos" (isso é domo).

**As três protagonistas:**
- **Covariância** (`∝ XᵀX`): sempre simétrica e PSD, porque `xᵀ(XᵀX)x = ‖Xx‖² ≥ 0`
  (comprimento² nunca é negativo — é o produto interno consigo mesmo).
- **Hessiana** (curvatura de uma função): positiva definida = tigela = **mínimo
  garantido**. É o que fecha o "por que é mínimo?" dos mínimos quadrados — a rampa
  de erro é sempre tigela/calha, nunca sela.
- **Kernels** (SVM): PSD por construção.

**ML:** `np.linalg.eigvalsh(A)` (autovalores de simétrica; todos >0 → PD).

---

# 11. SVD — o boss final ✅

**A pergunta:** autovetores só se comportam bem em matrizes quadradas, mas dados são
**retangulares** (200 casas × 3 features). A SVD responde: **toda** matriz, de
qualquer formato, é `A = U Σ Vᵀ` — **girar → esticar → girar**.

**Os três pedaços (tudo que já sei):**
- **`Vᵀ`** = 1ª rotação (ortogonal, `VᵀV=I`, só reorienta).
- **`Σ`** = esticamento (diagonal; os **valores singulares** σ = quanto estica cada
  direção — primos dos autovalores, mas pra qualquer matriz).
- **`U`** = 2ª rotação (reposiciona na saída).

**Imagem geométrica:** toda matriz transforma um **círculo numa elipse**. Vᵀ gira o
círculo, Σ estica em elipse (os σ são os semi-eixos), U gira a elipse. Guardar:
**qualquer matriz = girar → esticar → girar.**

**A ponte de ouro (aqui tudo se junta):** de onde saem os pedaços? Da `XᵀX` — que eu
provei ser **simétrica** e **PSD**. Então:
- os **autovetores de `XᵀX`** são as colunas de **V**;
- os **autovalores de `XᵀX`** são os **valores singulares ao quadrado** (`σ² = λ`);
- como `XᵀX` é **PSD** (todos λ ≥ 0), a raiz `σ = √λ` **sempre existe** — é por isso
  que a positividade da tigela importa aqui!

Ou seja: **a SVD é a decomposição espectral da `XᵀX`, reorganizada pra funcionar na
`X` retangular.** Cada peça que construí (transposta, simétrica, PSD, autovetores
ortogonais) era um ingrediente dela.

**O clímax — SVD é PCA:** PCA = direções de maior variância = autovetores da
covariância = autovetores de `XᵀX` = o **V** da SVD. Fazer PCA é fazer a SVD de X e
olhar V (direções) e Σ (variância de cada). Na prática usa-se SVD direto (mais
estável), por isso o `PCA` do sklearn é SVD por baixo.

**Redução de dimensão / compressão:** os σ vêm ordenados do maior pro menor. Ficar
com os `k` maiores e descartar o resto = **PCA / compressão de imagem /
embeddings**. Guardar o essencial, jogar fora o ruído.

**ML:** `U, S, Vt = np.linalg.svd(X)`.

---

## Mapa de interligações

| Conceito | O que responde | Liga com |
|---|---|---|
| Transformação | como o espaço é movido? | colunas = destino da base |
| Multiplicação | como compor? | ordem importa |
| Determinante | quanto a área escala? | det=0 = colapso |
| Posto / espaço-coluna | quantas dimensões sobrevivem? | span, imagem |
| Inversa | dá pra desfazer? | triângulo com det/posto |
| Transposta | espelhar e medir | dualidade → produto interno |
| Produto interno | quanto alinhamento? | sobreposição |
| Ortogonalidade | há sobreposição? | zero = não-redundante |
| Projeção | qual a melhor sombra? | erro ⊥ → regressão |
| Simétrica / PSD | qual a forma de xᵀAx? | tigela → mínimo garantido |
| Autovetores | quais direções só esticam? | PCA, espectral |
| SVD | como decompor qualquer matriz? | XᵀX, PCA, compressão |

---

## Mini-checks (Feynman — me forçar a explicar)

1. Por que `Wx+b` não é linear? → não preserva a origem; é afim.
2. Se `det=0`, por que não há inversa? → colapsou, perdeu informação.
3. Se `xᵀy=0`, o que significa? → ortogonais; zero pistas sobre a direção do outro.
4. Por que o erro da projeção é ⊥ ao subespaço? → se sobrasse componente dentro,
   dava pra ajustar e reduzir o erro.
5. Por que "derivada zero" e "erro perpendicular" dão o mesmo w? → mesma física: a
   derivada de ‖e‖² **é** o Xᵀe; rampa plana = sombra perpendicular.
6. Por que todos λ>0 → tigela? → cada λ é o quanto sobe/desce a pista; todos + →
   sobe em toda direção → mínimo.
7. Por que a SVD existe pra `X` retangular? → vem dos autovetores da `XᵀX`
   (simétrica e PSD); `σ=√λ` existe porque λ≥0.

---

## Próximos passos

1. **Drillar autovalores:** `det(A − λI) = 0` (o único pendente da álgebra linear).
2. **Cálculo / Otimização:** gradiente, Jacobiano, Hessiana, gradient descent,
   Lagrange (SVM). ← o gradiente do erro que já vi na projeção vira gradient descent aqui.
3. **Probabilidade / Estatística:** Bayes, distribuições, MLE, viés-variância,
   testes de hipótese (base de conformal prediction).
4. **Aplicar** — river-lifelines: a projeção vira a regressão, o PCA reduz as
   features. É onde a matemática vira habilidade.

---

> **Fechamento:** a ordem é **transformar → medir → separar → projetar → decompor**,
> e o próximo capítulo é **otimizar** (cálculo). A álgebra linear está
> essencialmente inteira — falta só drillar o `det(A−λI)=0` e, principalmente,
> **usar tudo isso em dado real.**
