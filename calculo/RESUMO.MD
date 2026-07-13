# Cálculo & Otimização para ML — o meu cérebro

> Segundo bloco de base, construído no mesmo estilo do de álgebra linear: cada
> conceito só entrou depois de eu **explicar de volta** e a gente afinar. Com as
> minhas palavras nos trechos que eu mesmo construí, e conectado com a álgebra
> linear e com ML — porque os dois blocos não são separados, são um motor só.
>
> Fórmulas em texto (não LaTeX), pra renderizar em qualquer lugar.

---

## A grande imagem (por que este bloco existe)

> **A álgebra linear me deu o *espaço* (vetores, projeção, autovetores, a tigela).
> O cálculo me dá o *movimento*: como descer a rampa de erro e como uma mudança se
> propaga por uma cadeia de funções.**

Treinar um modelo = achar os parâmetros que dão o menor erro. Cálculo é a
ferramenta que encontra esse fundo.

---

## O fio condutor

> **derivada (inclinação) → regra da cadeia (backprop) → gradiente (a bússola
> vetorial) → gradient descent (descer a rampa) → Lagrange (descer com restrição).**

---

## Status

**✅ Fechado:** derivada · regra da cadeia · gradiente · convexidade/pontos críticos ·
Hessiana (vem do bloco da tigela) · gradient descent e SGD · **multiplicadores de Lagrange**.

**🟡 Tocado, não drillado:** o **Jacobiano** (o "gradiente" de uma função que
devolve vários números de uma vez; generaliza o gradiente pra saídas vetoriais).
Fecha rápido quando aparecer.

---

# 1. Derivada — a inclinação

**A pergunta que ela responde:** se eu mexer *um tiquinho* na entrada, a saída muda
quanto? É uma **taxa de variação** = a **inclinação** da curva num ponto.

- subida íngreme → derivada grande positiva
- descida íngreme → derivada grande negativa
- plano (fundo de vale ou topo) → derivada **zero**

**O que mais importa — derivada zero = ponto crítico.** No fundo de um vale, a
inclinação é zero (a curva para de descer). No topo de um morro, idem. Por isso, pra
achar o menor erro, a gente **deriva e iguala a zero** — está procurando o fundo.

**Ajustes que travei:**
- A derivada **não é uma reta** e não "minimiza" nada — ela é o **inclinômetro** (a
  ferramenta que mede a inclinação). Quem minimiza sou eu, usando ela pra achar onde
  a inclinação zera.
- Separar duas coisas: a **reta do modelo** (`ax+b`, o que eu ajusto) da **rampa de
  erro** (a superfície do erro *sobre os parâmetros*). No eixo horizontal da rampa
  estão os **parâmetros (a, b)**, não os dados — os dados são fixos; o que eu mexo é
  o a e o b.
- Derivada zero diz "aqui é plano", mas não diz se é vale, morro ou sela. Pra
  confirmar que é **mínimo**, olho a **curvatura** (2ª derivada / Hessiana).

**Mecânica (só pra fechar):** `f'(x) = limite quando h→0 de [f(x+h) − f(x)] / h` —
a inclinação instantânea (subida/distância num ponto exato). Em ML o computador
calcula (autograd); o que importa é entender que é a inclinação.

**🔗 Liga com álgebra linear:** derivada zero **+** Hessiana positiva definida (a
**tigela** que fechei) = mínimo garantido.

---

# 2. Regra da cadeia — as engrenagens

**A pergunta:** como uma mudança se propaga por uma **cadeia de funções**
(`x → g → y → f → z`)? Resposta: **multiplicando as taxas ao longo do caminho.**
```
dz/dx = dz/dy · dy/dx
```

**A intuição (minha) — engrenagens:** *se a engrenagem A faz a B girar 2× mais
rápido, e a B faz a C girar 3× mais rápido, então A faz a C girar 2×3 = 6× mais
rápido.* As razões se **multiplicam** ao longo da cadeia.

**Por que multiplica (e não soma):** cada elo **amplifica o que veio antes** — a
mudança "viaja" pela cadeia sendo escalada em cada função. Amplificação em cima de
amplificação é multiplicação. **Derivada = fator de amplificação.**

**🔗 Liga com ML — backprop (minhas palavras):**
> *Uma rede é uma cadeia de funções gigantesca em que cada uma alimenta a outra.
> Pra treinar, é preciso saber: se eu mexer um peso na primeira camada, o erro final
> muda quanto? Como cada função alimenta a outra, é preciso multiplicar todas de trás
> pra frente — e a isso se dá o nome de **backpropagation**.*

Backprop **é** a regra da cadeia aplicada numa cadeia profunda, calculada de trás
pra frente. E a "cadeia de funções" é a **composição de transformações** (as
camadas = multiplicação de matrizes) que vi na álgebra linear.

---

# 3. Gradiente — a bússola vetorial

**O problema:** funções de ML têm **várias** entradas (o a e o b juntos, ou milhões
de pesos). "Inclinação pra qual direção?" Uma derivada só não basta.

**A peça:** a **derivada parcial** — derivar em relação a *uma* entrada segurando as
outras fixas ("se eu mexer só o a, o erro muda quanto?"). O **gradiente** empacota
todas num **vetor**:
```
∇f = ( ∂f/∂a , ∂f/∂b , ... )
```

**A intuição (minhas palavras):**
> *O gradiente aponta, com precisão cirúrgica, para a direção da subida mais íngreme.
> Se eu der um passo na direção que ele aponta, subo o mais rápido possível. O
> tamanho do vetor diz quão íngreme está a ladeira.*

**Corolário de ouro:** se `∇f` aponta pra subida mais íngreme, então `−∇f` aponta pra
**descida** mais íngreme. (É o próximo tópico.)

**Gradiente zero = mínimo (minhas palavras):**
> *No fundo, o gradiente vira o vetor zero porque não existe nenhuma direção de subida
> imediata. Pra qualquer lado que eu ande, o terreno sobe. O chão ficou nivelado em
> todas as direções ao mesmo tempo.*

Ajuste de rigor: gradiente zero marca um **ponto crítico** (vale, morro ou sela) —
a curvatura (Hessiana / tigela vs. domo vs. sela) confirma qual.

**🔗 A conexão mais linda com álgebra linear:** achar o mínimo é resolver `∇f = 0`.
Nos mínimos quadrados, `∇f = 0` **é exatamente** o `XᵀX ŵ = Xᵀb` que projetei. O
gradiente do erro igualado a zero = a equação normal. Cálculo e projeção são a mesma
coisa. E o gradiente é um vetor, então tudo volta: direção, produto interno (a
inclinação numa direção qualquer = `∇f · direção`), magnitude.

---

# 4. Gradient Descent (e SGD) — descer a rampa

**O problema:** em modelos grandes (redes com milhões de pesos) **não existe fórmula
fechada** pra `∇f = 0` — a paisagem é complexa demais pra isolar os parâmetros. Então
não dá pra resolver de uma vez, como no mínimos quadrados.

**A ideia (minhas palavras):**
> *É como se tudo estivesse escuro e eu precisasse descer a montanha. Uso a inclinação
> sob o pé, vejo onde está mais inclinado pra baixo, dou um passo, e vou fazendo isso
> até achar o fundo.*

Não preciso enxergar o mínimo — só saber, localmente, pra onde é ladeira abaixo:
**`−∇f`**.

**A mecânica (uma linha):**
```
novo_parâmetro = parâmetro_atual − η · ∇f
```
- `∇f` = direção da subida · `−` = vira pra descida · `η` (eta) = **tamanho do passo**
  (learning rate: grande = rápido mas pode pular o fundo; pequeno = seguro mas lento).

Repete milhares de vezes até o gradiente ficar ~zero (chegou perto do fundo).

**SGD (minhas palavras):** em vez de usar todos os dados a cada passo, usa um
**pedacinho aleatório (mini-batch)**:
> *1) começa com pesos aleatórios; 2) olha um mini-batch; 3) calcula o gradiente
> negativo (direção que diminui o erro); 4) dá um passo minúsculo (ajustando os pesos
> via backprop); 5) repete milhões de vezes.*

**🔗 Liga com ML:** treinar QUALQUER modelo = descer a rampa de erro. Backprop calcula
o gradiente (regra da cadeia); gradient descent dá o passo (`−η∇f`). Os dois juntos
treinam tudo. **Por que precisa dele:** porque na maioria dos modelos reais não há
fórmula fechada pro mínimo — só dá pra chegar lá caminhando.

---

# 5. Multiplicadores de Lagrange — descer com restrição

**O problema:** achar o mínimo **sujeito a uma regra** — os pesos do portfólio somam
100%, os pontos ficam do lado certo (SVM). A restrição te prende numa **trilha**
dentro da paisagem; você só pode andar sobre ela.

**A intuição (minhas palavras):**
> *O mínimo restrito acontece na **tangência** porque é o ponto de virada — é a curva
> de nível mais baixa que a trilha consegue sequer encostar antes de ser forçada a
> subir a montanha de novo.*

Enquanto a trilha **cruza** curvas de nível, dá pra escorregar e descer mais. Só onde
ela **tangencia** (encosta sem cruzar) é que andar pra qualquer lado não abaixa mais.

**Onde o gradiente entra:** no ponto de tangência, a curva de nível do erro e a trilha
ficam paralelas → seus gradientes ficam **paralelos** (mesma reta):
```
∇f = λ · ∇g
```

**O λ (minhas palavras, com ajuste):**
> *O λ é um número corretor (constante de escala): a bússola da montanha (∇f) fica
> paralela à bússola da estrada (∇g), só com comprimentos diferentes — o λ iguala as
> duas. É pura poesia geométrica.*

Ajuste de rigor: os gradientes ficam **paralelos** (mesma reta), podendo apontar pro
mesmo sentido **ou oposto** — por isso o λ pode ser **positivo ou negativo** (ele
ajusta comprimento *e* sentido).

**O significado do λ:** mede **o quanto a restrição está te segurando** ("preço-sombra"
— quanto valeria relaxar a regra um tiquinho). λ grande = restrição cara; λ = 0 =
restrição não atrapalha.

**🔗 Liga com ML — SVM:** achar a fronteira de margem máxima é otimização com
restrição; Lagrange resolve, e os **vetores de suporte** são os pontos onde λ ≠ 0
(as restrições "ativas" que seguram a fronteira). E o SVM opera sobre **produtos
internos** — a peça da álgebra linear.

---

## Mapa de interligações

**Dentro do cálculo:**
derivada (inclinação) → regra da cadeia (propaga a inclinação por uma cadeia) →
gradiente (inclinação em várias dimensões) → gradient descent (usa `−∇f` pra descer)
→ Lagrange (desce preso a uma trilha).

**Cálculo ↔ Álgebra Linear:**
| Cálculo | encontra | Álgebra Linear |
|---|---|---|
| derivada zero (ponto crítico) | + | Hessiana positiva definida = **tigela** = mínimo |
| gradiente zero nos mínimos quadrados | é | **`XᵀX ŵ = Xᵀb`** (equação normal, projeção) |
| regra da cadeia (backprop) | treina | **composição de transformações** (as camadas) |
| gradiente | é um | **vetor** (direção, produto interno, magnitude) |
| Lagrange (SVM) | opera sobre | **produtos internos** |

**Cálculo ↔ ML:**
gradient descent/SGD = motor de treino de todo modelo · backprop = regra da cadeia
em escala · Lagrange = SVM, regularização, portfólio · a rampa de erro = a loss
surface sobre os parâmetros.

---

## Mini-checks (Feynman)

1. Por que derivada zero marca mínimo/máximo? → é onde a inclinação some (fundo/topo).
2. Na rampa de erro, o que está no eixo horizontal? → os **parâmetros** (a, b), não os dados.
3. Por que as derivadas se multiplicam numa cadeia? → cada elo amplifica o anterior;
   amplificações se compõem multiplicando.
4. Por que `−∇f` é a direção de treino? → é a descida mais íngreme da rampa de erro.
5. Por que precisa de gradient descent se dá pra resolver `∇f=0`? → em modelos
   grandes não há fórmula fechada; só dá pra descer passo a passo.
6. Por que o mínimo restrito é na tangência? → cruzar curva de nível = dá pra
   melhorar; tangenciar = não há mais pra onde descer na trilha.
7. Como confirmar que um ponto crítico é mínimo? → curvatura: Hessiana positiva
   definida (tigela).

---

## Próximos passos

1. **Probabilidade / Estatística** — o último bloco de base (Bayes, distribuições,
   esperança/variância, MLE, viés-variância, testes de hipótese).
2. **Curso do IMPA** — amarra tudo, com a base pronta.
3. **river-lifelines (Passo 3)** — aplicar: a projeção vira regressão, o gradiente
   vira o treino. Onde a matemática vira habilidade.

Lacunas pequenas pra fechar quando aparecerem (não travar agora): **Jacobiano** a
fundo (quando ver saídas vetoriais), e no bloco anterior a mecânica do `det(A−λI)=0`.

---

> **Fechamento:** álgebra linear = o espaço; cálculo = o movimento. Juntos são o motor
> do ML: descer a rampa de erro (gradient descent) ajustando parâmetros, propagando a
> mudança por camadas (backprop), respeitando restrições (Lagrange), até o fundo da
> tigela — que a álgebra linear garante ser mínimo. Falta só a **probabilidade** pra
> base ficar completa.