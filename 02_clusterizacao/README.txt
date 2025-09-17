
GCC128 - IA | Trabalho Prático 02 - K-means (Iris)

Arquivos incluídos:
- km_hardcore.py               -> Implementação do KMeans do zero (hardcore)
- run_experiments.py           -> Script para rodar os experimentos básicos (k=3 e k=5)
- resumo_metricas.csv          -> Tabela com métricas (silhouette, inertia, iterações, tempo)
- pca1_bestk_<k>.png           -> Visualização PCA 1 componente (melhor k)
- pca2_bestk_<k>.png           -> Visualização PCA 2 componentes (melhor k)
- relatorio_kmeans.pdf         -> Relatório de até 1 página (PDF)
- README.txt                   -> Este arquivo

Como executar localmente:
1) Instale as dependências: scikit-learn, numpy, pandas, matplotlib.
2) Rode:  python run_experiments.py
3) Os gráficos e o PDF já estão gerados nesta pasta.

Observações importantes:
- O dataset Iris é carregado via sklearn.datasets; a variável alvo NÃO é usada.
- Os atributos foram padronizados (z-score) antes do agrupamento.
- O melhor k foi escolhido via maior silhouette (sklearn).
- Substitua "Autores: Nome1, Nome2" no PDF pelos seus nomes.
