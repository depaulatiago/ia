# ============================================================
# GCC128 - Inteligência Artificial
# Professores: Ahmed Ali Abdalla Esmin / Anna Paula Figueiredo
# Trabalho Prático 03 - MLPClassifier
# Alunos: Tiago de Paula Martins e Samuel Moreira Abreu
# Data: 10/10/2025
# ============================================================

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# ------------------------------------------------------------
# Função de avaliação
# ------------------------------------------------------------
def avaliar_base(dados, nome_base, caminho_pasta="imagens"):
    """Treina e avalia o MLPClassifier em uma base de dados, salvando a imagem da matriz de confusão."""
    X, y = dados.data, dados.target
    nomes_classes = dados.target_names

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Criação e treino do modelo
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10),
                        activation='relu',
                        solver='adam',
                        max_iter=1000,
                        random_state=42)

    inicio = time.time()
    mlp.fit(X_train, y_train)
    tempo = time.time() - inicio

    # Predição
    y_pred = mlp.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Resultados - Base {nome_base} ===")
    print(f"Acurácia: {acc:.4f}")
    print(f"Tempo de treinamento: {tempo:.4f}s")
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred, target_names=nomes_classes))

    # Criação da pasta de imagens (se não existir)
    os.makedirs(caminho_pasta, exist_ok=True)

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title(f"Matriz de Confusão - {nome_base}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")

    # Caminho do arquivo de imagem
    nome_arquivo = f"matriz_confusao_{nome_base.lower()}.png"
    caminho_imagem = os.path.join(caminho_pasta, nome_arquivo)

    # Salvar figura (sobrescrevendo caso já exista)
    plt.tight_layout()
    plt.savefig(caminho_imagem)
    plt.close()

    print(f"📊 Matriz de confusão salva em: {caminho_imagem}")

    return acc, tempo


# ------------------------------------------------------------
# Execução principal
# ------------------------------------------------------------
if __name__ == "__main__":
    iris = load_iris()
    wine = load_wine()

    acc_iris, tempo_iris = avaliar_base(iris, "Iris")
    acc_wine, tempo_wine = avaliar_base(wine, "Wine")

    print("\n--- Comparação Geral ---")
    print(f"Iris -> Acurácia: {acc_iris:.4f} | Tempo: {tempo_iris:.4f}s")
    print(f"Wine -> Acurácia: {acc_wine:.4f} | Tempo: {tempo_wine:.4f}s")