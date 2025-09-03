# Deep Learning — Entregas e Relatório (Caio Ortega Boa)

???+ info inline end "Edição"
    **Autor:** *Caio Ortega Boa*  
    **Disciplina:** Deep Learning  
    **Período:** 2025.1  
    **Descrição:** Esta página reúne **todo o conteúdo produzido** (gráficos e análises) para as três atividades da disciplina.

---

## Sumário

- ✅ **Exercício 1:** Separabilidade em 2D (dados sintéticos gaussianos)  
- ✅ **Exercício 2:** Não-linearidade em 5D e projeção PCA → 2D  
- ✅ **Exercício 3:** Pré-processamento do *Spaceship Titanic* (Kaggle) para MLP com `tanh`

!!! tip "Como usar este material"
    - As **figuras** referenciadas devem estar no **mesmo diretório** desta página (ou ajuste os caminhos).  
    - Cada seção descreve **o que foi feito**, **por que foi feito** e traz as **respostas solicitadas** no enunciado.  
    - Os notebooks/códigos ficam no repositório e podem ser executados localmente.

---

## Entregas (Checklist)

- [x] Exercício 1 — Geração/visualização, análise e “sketch” de fronteiras  
- [x] Exercício 2 — Geração 5D, PCA (5D→2D) e análise de separabilidade  
- [x] Exercício 3 — Descrição do dataset, tratamento de faltantes, *encoding* e **escala apropriada para `tanh`**; histogramas antes/depois

---

## Exercício 1 — Class Separability em 2D

**Objetivo.** Explorar como a distribuição de quatro classes em 2D influencia a complexidade das fronteiras de decisão que uma rede neural precisaria aprender.

**Parâmetros utilizados (por classe).**  
Médias (μx, μy): **(2,3)**, **(5,6)**, **(8,1)**, **(15,4)**  
Desvios (σx, σy): **(0,8; 2,5)**, **(1,2; 1,9)**, **(0,9; 0,9)**, **(0,5; 2,0)**  
*Observação:* desvios por eixo ⇒ **elipses alinhadas aos eixos** (covariância diagonal, sem rotação).

**Visualizações.**  
![Scatter 2D](ex1_scatter.png)  
![Fronteiras desenhadas manualmente](ex1_lines.png)

**Análise e respostas.**
- **Distribuição e overlap:** Classes 0 e 1 apresentam sobreposição sobretudo no eixo vertical; a Classe 2 é mais compacta; a Classe 3 está deslocada à direita e mais alongada em \(y\).  
- **Uma fronteira linear simples separa tudo?** Não. Uma única linha não separa as quatro classes simultaneamente; mesmo com múltiplos hiperplanos lineares, há erros nas zonas de mistura.  
- **“Sketch” das fronteiras que a rede aprenderia:** Fronteiras **curvas** que contornam os aglomerados (principalmente nas interfaces 0↔1 e 1↔2) tendem a melhorar a separação em relação a limites estritamente lineares.

**Critérios atendidos.**  
Geração correta dos dados e *scatter* claro; análise de separabilidade e proposta de fronteiras coerentes.

---

## Exercício 2 — Não-linearidade em 5D + PCA (5D → 2D)

**Objetivo.** Criar dois grupos 5D com médias/covariâncias especificadas e visualizar em 2D via **PCA**.

**Configuração.**  
- **Classe A:** vetor de média nulo; covariâncias positivas entre algumas dimensões.  
- **Classe B:** vetor de média transladado (1,5 em todas as componentes); covariâncias com sinais distintos, alterando forma e orientação do grupo.

**Projeção e visualização.**  
![PCA 2D](ex2_pca.png)

**Análise e respostas.**
- **Relação entre as classes (projeção 2D):** Observa-se **mistura parcial**; as nuvens são alongadas e com **orientações diferentes**, refletindo as covariâncias distintas no 5D.  
- **Separabilidade linear:** A projeção não mostra separação linear perfeita; parte da informação separadora está em dimensões além das duas primeiras.  
- **Por que redes com não-linearidade?** As covariâncias induzem **fronteiras curvas** no espaço original; modelos lineares não capturam bem essa geometria. Uma MLP com ativações não lineares (ex.: `tanh`) é mais apropriada.

**Critérios atendidos.**  
Dados gerados com os parâmetros; PCA aplicada corretamente; análise de não-linearidade consistente.

---

## Exercício 3 — *Spaceship Titanic* (Kaggle): Pré-processamento para `tanh`

**Objetivo.** Preparar dados reais para uma MLP com `tanh`, assegurando entradas estáveis (centralizadas e/ou limitadas).

### Descrição do dataset

- **Alvo:** `Transported` — indica se o passageiro foi transportado para outra dimensão (binário).  
- **Numéricas (exemplos):** `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`, `CabinNum`, `Group`, `PaxInGroup`, `TotalSpend`.  
- **Categóricas (exemplos):** `HomePlanet`, `CryoSleep`, `Destination`, `VIP`, `CabinDeck`, `CabinSide`.  
- **Engenharia aplicada:**  
  - `Cabin` decomposta em `CabinDeck`, `CabinNum`, `CabinSide`;  
  - `PassengerId` decomposto em `Group`, `PaxInGroup`;  
  - criação de `TotalSpend` (soma das despesas);  
  - `Transported` convertido para 0/1.

### Faltantes

- **Numéricas:** imputação pela **mediana** (robusta a outliers; preserva a posição central).  
- **Categóricas:** imputação pela **moda** (mantém rótulos conhecidos; evita categorias artificiais).  
- **Justificativa:** estratégia simples e estável para *pipelines*, apropriada ao contexto de redes com `tanh`.

### Codificação de categóricas

- **One-Hot Encoding** com ignorância de rótulos desconhecidos em validação/teste, evitando falhas quando surgem categorias inéditas.

### Escala para `tanh`

- **Opção A — Padronização (z-score):** centra em 0 e ajusta desvio para 1, posicionando as entradas na região de maior sensibilidade da `tanh`.  
- **Opção B — Normalização para `[-1, 1]`:** atende ao requisito de intervalo estrito da rubrica.  
  - Observação: em validação/teste, valores fora do min/máx do treino podem ultrapassar `[-1,1]`; se necessário, aplicar *clipping*.  
  - Para colunas muito assimétricas (ex.: gastos), uma transformação **log1p** antes da normalização melhora a estabilidade.

### Visualizações do efeito da escala

**Antes da transformação**  
![Antes — Age e FoodCourt](ex3_hist_before.png)

**Depois da transformação** *(z-score ou `[-1,1]`, conforme a opção escolhida)*  
![Depois — Age e FoodCourt](ex3_hist_after.png)

**Critérios atendidos.**  
Carregamento e descrição corretos; tratamento de faltantes, codificação e escala apropriados para `tanh`; visualizações evidenciando o impacto das transformações.

---

## Critérios de Avaliação — Checklist de Evidências

**Exercício 1 (3 pts)**  
- [x] Dados gerados e *scatter* claro com rótulos/cores  
- [x] Análise de separabilidade e fronteiras propostas

**Exercício 2 (3 pts)**  
- [x] Dados 5D com parâmetros especificados  
- [x] PCA aplicada e *scatter* 2D apresentado  
- [x] Discussão sobre não-linearidade e adequação de MLP

**Exercício 3 (4 pts)**  
- [x] Dataset carregado e características descritas  
- [x] Pré-processamento completo (faltantes, *encoding*, escala justificada para `tanh`)  
- [x] Histogramas antes/depois mostrando o impacto da transformação

---

## Referências

[NumPy](https://numpy.org/) • [scikit-learn](https://scikit-learn.org/) • [Matplotlib](https://matplotlib.org/) • Dataset *Spaceship Titanic* (Kaggle) • Política acadêmica da disciplina
