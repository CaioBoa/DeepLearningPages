# Deep Learning — Multi-Layer Perceptron (MLP)

**Autor:** *Caio Ortega Boa*  
**Disciplina:** Deep Learning  
**Período:** 2025.1  
[Link do Repositório](https://github.com/CaioBoa/-MultiLayerPerceptrons)

---

## Sumário

- **Exercício 1** Cálculo Manual de uma MLP
- **MLP**: rede com camadas ocultas `tanh` e saída `softmax`  
- **Exercício 2:** Classificação binária  
- **Exercício 3:** Classificação multiclasse 
- **Exercício 4:** Classificação multiclasse (+ camadas ocultas)

---

## MLP

**Objetivo.** Implementar uma MLP **modular** em NumPy, suportando:
- **Entrada genérica** (`input_dim = n_features`);
- **N camadas ocultas** (lista de larguras), **ativação `tanh`** em todas as ocultas;
- **Camada de saída `softmax`** com `output_dim = n_classes` (funciona para **binário** com `K=2` e para **multiclasse**, `K>2`);
- **Loss:** *categorical cross-entropy* (com one-hot);
- **Otimização:** *Gradient Descent* puro;
- **Histórico de treino:** *loss* e *accuracy* por época.

**Fluxo do treino.**  
1. Inicialização dos parâmetros (pesos via Xavier, vieses em zero);  
2. *Forward* (tanh nas ocultas; softmax na saída);  
3. *Loss* (cross-entropy) e métricas;  
4. *Backward* (gradientes por regra da cadeia, com **dZ = (P − Y)/m** na saída softmax+CE);  
5. Atualização via *Gradient Descent*;  
6. Registro de *loss/acc* a cada época e avaliação final no *test set*.

```python
# mlp.py
from __future__ import annotations
from typing import List, Optional
import numpy as np

from utils import (
    tanh, dtanh_from_a,
    softmax, one_hot, cross_entropy, accuracy_score,
    xavier_init,
)

class MLP:
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [16, 16],
        output_dim: int = 2,        
        lr: float = 0.05,
        max_epochs: int = 500,
        batch_size: Optional[int] = None,
        random_state: Optional[int] = 42,
        track_history: bool = True,
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.track_history = track_history

        self.params_ = None
        self.loss_history_: List[float] = []
        self.acc_history_: List[float] = []

    # ---------- initialization ----------
    def _init_params(self, rng: np.random.Generator) -> None:
        layer_sizes = [self.input_dim] + self.hidden_layers + [self.output_dim]
        W, b = [], []
        for l in range(1, len(layer_sizes)):
            fan_in = layer_sizes[l-1]
            fan_out = layer_sizes[l]
            W_l = xavier_init(fan_in, fan_out, rng)
            b_l = np.zeros((fan_out, 1))
            W.append(W_l)
            b.append(b_l)
        self.params_ = {"W": W, "b": b}

    # ---------- forward ----------
    def _forward(self, X: np.ndarray):
        W, B = self.params_["W"], self.params_["b"]
        A = X.T  
        caches = [{"A": A}]  

        # hidden layers
        for l in range(len(self.hidden_layers)):
            Z = W[l] @ A + B[l]
            A = tanh(Z)
            caches.append({"Z": Z, "A": A})

        # output layer (softmax)
        ZL = W[-1] @ A + B[-1]
        P = softmax(ZL, axis=0)
        caches.append({"Z": ZL, "A": P})
        return caches, P.T

    # ---------- backward ----------
    def _backward(self, caches, y: np.ndarray):
        W = self.params_["W"]
        L = len(W)
        m = y.shape[0]

        A0 = caches[0]["A"]
        A_list = [A0] + [c["A"] for c in caches[1:]]

        Y = one_hot(y.reshape(-1), self.output_dim).T
        P = A_list[-1]

        dZ = (P - Y) / m
        dW = [None] * L
        dB = [None] * L

        # última camada
        A_prev = A_list[-2]
        dW[L-1] = dZ @ A_prev.T
        dB[L-1] = np.sum(dZ, axis=1, keepdims=True)

        # ocultas
        for l in reversed(range(L-1)):
            dA = W[l+1].T @ dZ
            A_l = A_list[l+1]
            dZ = dA * dtanh_from_a(A_l)

            A_prev = A_list[l]
            dW[l] = dZ @ A_prev.T
            dB[l] = np.sum(dZ, axis=1, keepdims=True)

        return dW, dB

    # ---------- update ----------
    def _update(self, dW, dB, lr: float) -> None:
        for l in range(len(self.params_["W"])):
            self.params_["W"][l] -= lr * dW[l]
            self.params_["b"][l] -= lr * dB[l]

    # ---------- fit ----------
    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        self._init_params(rng)

        m = X.shape[0]
        batch_size = self.batch_size or m

        for epoch in range(1, self.max_epochs + 1):
            idx = rng.permutation(m)
            X_shuf = X[idx]
            y_shuf = y[idx]

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]

                caches, _ = self._forward(Xb)
                dW, dB = self._backward(caches, yb)
                self._update(dW, dB, self.lr)

            if self.track_history:
                P_full = self.predict_proba(X)
                Y_full = one_hot(y, self.output_dim)
                loss = cross_entropy(Y_full, P_full)
                y_pred = np.argmax(P_full, axis=1)
                acc = accuracy_score(y, y_pred)
                self.loss_history_.append(loss)
                self.acc_history_.append(acc)

        return {"epochs_run": self.max_epochs}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, P = self._forward(X)
        return P

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        W, B = self.params_["W"], self.params_["b"]
        A = X.T
        for l in range(len(self.hidden_layers)):
            A = tanh(W[l] @ A + B[l])
        ZL = W[-1] @ A + B[-1]
        return ZL.T

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = self.predict_proba(X)
        return np.argmax(P, axis=1)
```

**Funções Auxíliares**
```python
# utils.py
from __future__ import annotations
import numpy as np

# -----------------------------
# Ativações e derivadas
# -----------------------------
def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def dtanh_from_a(a: np.ndarray) -> np.ndarray:
    return 1.0 - a**2

def softmax(Z: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Z: (K, m) -> aplica softmax por coluna (axis=0).
    Retorna prob. por classe, colunas somam 1.
    """
    Z_shift = Z - np.max(Z, axis=axis, keepdims=True)
    e = np.exp(Z_shift)
    return e / np.sum(e, axis=axis, keepdims=True)

# -----------------------------
# Loss e métricas
# -----------------------------
def bce_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

def cross_entropy(y_true_oh: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    """
    y_true_oh: (m, K) one-hot
    y_prob   : (m, K) probabilidades (softmax)
    """
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(np.sum(y_true_oh * np.log(y_prob), axis=1)))

def accuracy_score(y_true: np.ndarray, y_pred_labels: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred_labels))

# -----------------------------
# Split 80/20
# -----------------------------
def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    m = X.shape[0]
    idx = rng.permutation(m)
    m_test = int(np.floor(test_size * m))
    test_idx = idx[:m_test]
    train_idx = idx[m_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# -----------------------------
# Inicializações
# -----------------------------
def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, size=(fan_out, fan_in))

# -----------------------------
# Helpers
# -----------------------------
def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    """
    y: (m,) com rótulos inteiros [0..K-1]
    retorna: (m, K) one-hot
    """
    m = y.shape[0]
    Y = np.zeros((m, K), dtype=float)
    Y[np.arange(m), y.astype(int)] = 1.0
    return Y
```

**Pré-processamento.**  
- **MinMax [-1, 1]** nos atributos, para combinar com ativação **tanh** nas ocultas.

```python
from __future__ import annotations
import numpy as np

class MinMaxScaler:
    """
    Escala os dados para o intervalo [-1, 1].
    """
    def __init__(self):
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler não ajustado. Chame fit() antes de transform().")
        
        # normaliza para [0, 1]
        X_norm = (X - self.min_) / (self.max_ - self.min_ + 1e-12)
        # reescala para [-1, 1]
        return 2.0 * X_norm - 1.0
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
```

**Geração de Dados**
```python
from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

def make_varying_classification(
    n_samples: int,
    n_classes: int,
    n_features: int,
    clusters_per_class,
    class_sep: float = 1.2,           
    flip_y: float = 0.2,
    random_state: int = 42,
    shuffle: bool = False,
):
    """
    Gera dados sintéticos com nº de clusters variável por classe,
    usando make_classification de forma simplificada.

    - Divide amostras de forma uniforme entre as classes
    - n_informative = n_features
    - n_redundant = 0
    - class_sep e flip_y ajustáveis
    """
    clusters_per_class = list(clusters_per_class)
    if n_classes < 2 or len(clusters_per_class) != n_classes:
        raise ValueError("clusters_per_class deve ter n_classes elementos e n_classes >= 2.")

    # Divide amostras uniformemente
    base = n_samples // n_classes
    counts = [base] * n_classes
    for i in range(n_samples - base * n_classes):
        counts[i] += 1

    X_parts, y_parts = [], []
    for c, m_c in enumerate(counts):
        seed_c = (random_state + 10007 * (c + 1)) % (2**31 - 1)
        X_c, _ = make_classification(
            n_samples=m_c,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,                # make_classification exige >=2
            n_clusters_per_class=clusters_per_class[c],
            weights=[1.0, 0.0],         # força só uma classe
            class_sep=class_sep,
            flip_y=flip_y,
            shuffle=True,
            random_state=seed_c,
        )
        X_parts.append(X_c)
        y_parts.append(np.full(m_c, c, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    if shuffle:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(X.shape[0])
        X, y = X[idx], y[idx]

    return X, y

def plot_classification_data(X: np.ndarray, y: np.ndarray, title: str = "Synthetic Data"):
    """
    Plota os dados 2D gerados por make_varying_classification.
    
    Parâmetros:
      X : np.ndarray (n_samples, 2) -> features
      y : np.ndarray (n_samples,)   -> rótulos (0, 1, ..., n_classes-1)
      title : título opcional do gráfico
    """
    if X.shape[1] != 2:
        raise ValueError("O plot só funciona para n_features=2.")
    
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", edgecolor="k", s=40, alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
```

---

## Exercício 2 — Classificação Binária

**Objetivo.** Gerar um conjunto de dados **binário** (2 classes), com 2 *features* e **clusters assimétricos** (1 classe com 2 agrupamentos e a outra com 1 agrupamento), e treinar uma **MLP do zero** para classificá-lo.

### Especificação dos Dados
- **Amostras:** 1000  
- **Classes:** 2  
- **Features:** 2 (para visualização 2D)  
- **Clusters por classe:** `[2, 1]` (obtido combinando subconjuntos com `make_classification`)  
- **Informativas:** `n_informative = n_features`  
- **Redundantes:** `n_redundant = 0`  
- **Separabilidade:** `class_sep ≈ 1.2` (ajustado para ser desafiador, porém separável)  
- **Ruído de rótulo:** `flip_y ≈ 0.0`  
- **Reprodutibilidade:** `random_state` fixo

### Pipeline
- **Split:** 80% treino / 20% teste (embaralhado, *random_state* fixo);  
- **Scaler:** **MinMax [-1,1]** ajustado no treino e aplicado no teste;  
- **MLP:**  
  - Entrada: `input_dim = 2`;  
  - Ocultas: `[16, 16]` com ativação **tanh**;  
  - Saída: `output_dim = 2` (**softmax**);  
  - *Loss:* **cross-entropy** categórica (one-hot);  
  - *Optimizer:* **Gradient Descent**;  
  - *Batch size:* 64;  
  - *Épocas:* 500.

### Visualização
- **Dados Gerados:** dispersão 2D com classes bem definidas, porém não linearmente separáveis por uma única linha.  
- **Histórico:** gráfico da *loss* ao longo das épocas, evidenciando convergência suave.  

### Resultados
- **Train Loss:** ~0.23  
- **Train Accuracy:** ~0.91  
- **Test Loss:** ~0.27  
- **Test Accuracy:** ~0.90  

### Análise dos Resultados
- A MLP treinada do zero foi capaz de separar classes não linearmente separáveis.  
- O uso da ativação **tanh** nas camadas ocultas contribuiu para mapear regiões complexas do espaço de decisão.  
- A saída com **softmax** + *cross-entropy* garantiu treinamento estável e convergência.  
- A rede conseguiu generalizar bem, com desempenho semelhante em treino e teste.  

---

## Exercício 3 — Classificação Multiclasse (3 Classes)

**Objetivo.** Reutilizar exatamente a mesma implementação da MLP, alterando apenas parâmetros de saída e da geração de dados, para classificar um problema **multiclasse com 3 classes** e 4 *features*.

### Especificação dos Dados
- **Amostras:** 1500  
- **Classes:** 3  
- **Features:** 4  
- **Clusters por classe:** `[2, 3, 4]` (obtidos separadamente e combinados)  
- **Informativas:** `n_informative = 4`  
- **Redundantes:** `n_redundant = 0`  
- **Separabilidade:** moderada, com sobreposição entre alguns clusters.  
- **Split:** 80/20 treino/teste  

### Pipeline
- **Scaler:** **MinMax [-1,1]** em 4 *features*.  
- **MLP:**  
  - Entrada: `input_dim = 4`;  
  - Ocultas: `[16, 16]` com ativação **tanh**;  
  - Saída: `output_dim = 3` (**softmax**);  
  - *Loss:* cross-entropy categórica (one-hot);  
  - *Épocas:* 500.  

### Resultados
- **Train Accuracy:** ~0.85  
- **Test Accuracy:** ~0.82  
- **Histórico de Loss:** mostra convergência estável, sem overfitting evidente.  

### Análise dos Resultados
- O mesmo código da MLP binária foi reutilizado com sucesso, apenas ajustando `output_dim` e o *loss*.  
- A rede conseguiu lidar com a maior complexidade, separando múltiplas classes em um espaço de 4 dimensões.  
- Apesar da sobreposição entre clusters, o modelo atingiu boa generalização.  
- O resultado evidencia a flexibilidade da arquitetura para diferentes problemas.  

---
