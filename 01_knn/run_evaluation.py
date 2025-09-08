# run_evaluation.py
# Executa a avaliação para ks = {1,3,5,7} tanto para o "hardcore" quanto sklearn,
# gera as matrizes de confusão (PDF), o metrics.csv e um relatório de 1 página (PDF).

import os, time, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

from knn_hardcore import knn_predict
from knn_sklearn import build_knn

BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

def plot_confusion(cm, labels, title, save_path):
    # regra: matplotlib puro, 1 plot e sem cores fixas
    plt.figure()
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def make_report_text(results_hardcore, results_sklearn, ks):
    lines = []
    lines.append("GCC128 – Inteligência Artificial | Trabalho Prático 01 – Classificação KNN")
    lines.append("Alunos: Tiago de Paula Martins e Samuel Moreira Abreu")
    lines.append("Base: Iris (150 amostras, 3 classes). Split estratificado 70/30, padronização z-score.")
    lines.append("Ks avaliados: {1, 3, 5, 7}")
    lines.append("")

    lines.append("Resultados (acurácia | precisão(macro) | revocação(macro) | tempo[s]):")
    for label, res in [("Hardcore", results_hardcore), ("Sklearn", results_sklearn)]:
        lines.append(f"\n{label}:")
        for k in ks:
            m = res[k]["metrics"]
            lines.append(f"  k={k}: {m['accuracy']:.3f} | {m['precision_macro']:.3f} | {m['recall_macro']:.3f} | {m['time_sec']:.6f}")

    # Melhor por acurácia
    best_src, best_k, best_acc = None, None, -1.0
    for label, res in [("Hardcore", results_hardcore), ("Sklearn", results_sklearn)]:
        for k in ks:
            acc = res[k]["metrics"]["accuracy"]
            if acc > best_acc:
                best_acc = acc
                best_src, best_k = label, k

    lines.append("\nAnálise de Desempenho (resumo):")
    lines.append(f"- Melhor configuração observada: {best_src} com k={best_k} (acurácia={best_acc:.3f}).")
    lines.append("- As métricas tendem a ser semelhantes entre as abordagens, pois o algoritmo é o mesmo;")
    lines.append("  diferenças vêm de empates/votos e otimizações internas.")
    lines.append("- Tempo: sklearn tende a ser mais otimizado em cenários maiores; em Iris as diferenças são pequenas.")
    lines.append("- Conclusão: a implementação 'hardcore' é adequada para fins didáticos; para produção/escala,")
    lines.append("  a versão de biblioteca é mais conveniente e robusta.")

    return "\n".join(lines)

def save_report_pdf(report_text, save_path):
    # 1 página PDF com texto monoespaçado
    plt.figure(figsize=(8.27, 11.69))  # A4
    plt.axis('off')
    wrapped = textwrap.fill(report_text, width=90, replace_whitespace=False)
    plt.text(0.03, 0.97, wrapped, va='top', family='monospace')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()

def main():
    ks = [1,3,5,7]
    iris = load_iris()
    X = iris.data.astype(float)
    y = iris.target.astype(int)
    labels = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std  = scaler.transform(X_test)

    results_hardcore, results_sklearn = {}, {}

    # Hardcore
    for k in ks:
        t0 = time.perf_counter()
        y_pred = knn_predict(X_test_std, X_train_std, y_train, k=k)
        elapsed = time.perf_counter() - t0
        cm = confusion_matrix(y_test, y_pred)
        metrics = evaluate(y_test, y_pred)
        metrics["time_sec"] = elapsed
        results_hardcore[k] = {"confusion_matrix": cm.tolist(), "metrics": metrics}
        plot_confusion(cm, labels, f"KNN Hardcore (k={k})", os.path.join(FIG_DIR, f"cm_hardcore_k{k}.pdf"))

    # Sklearn
    for k in ks:
        model = build_knn(k)
        t0 = time.perf_counter()
        model.fit(X_train_std, y_train)
        y_pred = model.predict(X_test_std)
        elapsed = time.perf_counter() - t0
        cm = confusion_matrix(y_test, y_pred)
        metrics = evaluate(y_test, y_pred)
        metrics["time_sec"] = elapsed
        results_sklearn[k] = {"confusion_matrix": cm.tolist(), "metrics": metrics}
        plot_confusion(cm, labels, f"KNN Sklearn (k={k})", os.path.join(FIG_DIR, f"cm_sklearn_k{k}.pdf"))

    # Save metrics CSV
    rows = []
    for src, res in [("hardcore", results_hardcore), ("sklearn", results_sklearn)]:
        for k in ks:
            m = res[k]["metrics"]
            rows.append({
                "classifier": src, "k": k,
                "accuracy": m["accuracy"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "time_sec": m["time_sec"],
            })
    df = pd.DataFrame(rows).sort_values(["classifier","k"])
    df.to_csv(os.path.join(BASE_DIR, "metrics.csv"), index=False)

    # Generate report PDF
    report_text = make_report_text(results_hardcore, results_sklearn, ks)
    save_report_pdf(report_text, os.path.join(BASE_DIR, "relatorio_knn.pdf"))

if __name__ == "__main__":
    main()