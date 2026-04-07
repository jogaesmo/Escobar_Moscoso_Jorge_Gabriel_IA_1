
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import warnings
warnings.filterwarnings('ignore')


# 1. CARGA DE DATOS

df = pd.read_csv('credit_approval_processed.csv')

X = df.drop(columns=['approved'])
y = df['approved']

print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Variable objetivo: approved  |  Balance: {y.value_counts().to_dict()}")

# 2. DIVISIÓN DE DATOS — 75% entrenamiento / 25% prueba
#    stratify=y para mantener el balance de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nEntrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba:        {X_test.shape[0]} muestras")
print(f"Balance en prueba: {y_test.value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────
# 3. MODELOS CANDIDATOS
# ─────────────────────────────────────────────────────────
modelos = {
    'Logistic Regression':     LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'Decision Tree (max_d=4)': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest':           RandomForestClassifier(
                                   n_estimators=200,
                                   max_depth=10,
                                   min_samples_leaf=3,
                                   random_state=42,
                                   n_jobs=-1),
}

resultados = {}
print("\n" + "="*70)
print(f"{'Modelo':<30} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("="*70)

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred  = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec  = recall_score(y_test, pred)
    f1   = f1_score(y_test, pred)
    cv   = cross_val_score(modelo, X_train, y_train, cv=5, scoring='f1').mean()
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    resultados[nombre] = dict(modelo=modelo, pred=pred, proba=proba,
                               acc=acc, prec=prec, rec=rec, f1=f1,
                               cv_f1=cv, fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    print(f"{nombre:<30} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {roc_auc:>9.4f}")

print("="*70)

# 4. MEJOR MODELO
mejor_nombre = max(resultados, key=lambda k: resultados[k]['f1'])
mejor = resultados[mejor_nombre]
print(f"\n★ Mejor modelo: {mejor_nombre}")
print(f"  Accuracy  = {mejor['acc']:.4f}")
print(f"  Precision = {mejor['prec']:.4f}")
print(f"  Recall    = {mejor['rec']:.4f}")
print(f"  F1        = {mejor['f1']:.4f}")
print(f"  ROC-AUC   = {mejor['roc_auc']:.4f}")
print(f"  CV F1     = {mejor['cv_f1']:.4f}  (sin overfitting si ≈ F1 test)")
print(f"\nReporte completo:\n")
print(classification_report(y_test, mejor['pred'],
                             target_names=['Rechazado', 'Aprobado']))

# 5. VISUALIZACIONES
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(f'Evaluación del Modelo de Clasificación\n{mejor_nombre} — Credit Approval',
             fontsize=14, fontweight='bold')

# 5.1 Matriz de Confusión
ax = axes[0, 0]
cm = confusion_matrix(y_test, mejor['pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Rechazado', 'Aprobado'],
            yticklabels=['Rechazado', 'Aprobado'],
            linewidths=1, linecolor='white',
            annot_kws={'size': 16, 'weight': 'bold'})
ax.set_xlabel('Predicho', fontweight='bold')
ax.set_ylabel('Real', fontweight='bold')
ax.set_title('Matriz de Confusión')

# 5.2 Curva ROC — todos los modelos
ax = axes[0, 1]
colors = ['steelblue', 'seagreen', 'darkorange']
for (nombre, res), color in zip(resultados.items(), colors):
    etiqueta = nombre.replace(' Regression', '').replace(' (max_d=4)', '')
    ax.plot(res['fpr'], res['tpr'], lw=2, color=color,
            label=f'{etiqueta} (AUC={res["roc_auc"]:.3f})')
ax.plot([0, 1], [0, 1], '--', color='grey', lw=1.5)
ax.set_xlabel('Tasa Falsos Positivos')
ax.set_ylabel('Tasa Verdaderos Positivos')
ax.set_title('Curva ROC')
ax.legend(fontsize=9)

# 5.3 Comparativa de métricas
ax = axes[1, 0]
nombres_cortos = ['Logistic\nReg.', 'Dec.\nTree', 'Random\nForest']
metricas = {
    'Accuracy':  [resultados[n]['acc']  for n in resultados],
    'Precision': [resultados[n]['prec'] for n in resultados],
    'Recall':    [resultados[n]['rec']  for n in resultados],
    'F1':        [resultados[n]['f1']   for n in resultados],
}
x = np.arange(3); w = 0.2
for i, (metrica, vals) in enumerate(metricas.items()):
    ax.bar(x + i*w - 0.3, vals, w, label=metrica, alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(nombres_cortos)
ax.set_ylabel('Valor'); ax.set_title('Comparación de Métricas')
ax.set_ylim(0, 1.1); ax.legend(fontsize=9)

# 5.4 F1 Test vs CV — detección de overfitting
ax = axes[1, 1]
f1_test = [resultados[n]['f1']    for n in resultados]
f1_cv   = [resultados[n]['cv_f1'] for n in resultados]
x = np.arange(3); w = 0.35
b1 = ax.bar(x - w/2, f1_test, w, label='F1 Test',      color='steelblue', alpha=0.9)
b2 = ax.bar(x + w/2, f1_cv,   w, label='F1 CV (Train)', color='seagreen',  alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(nombres_cortos)
ax.set_ylabel('F1 Score'); ax.set_title('F1 Test vs CV — Detección de Overfitting')
ax.set_ylim(0, 1.1); ax.legend()
for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('resultados_clasificacion.png', dpi=150, bbox_inches='tight')
print("\nGráfico guardado: resultados_clasificacion.png")
plt.show()

# Importancia de variables (Random Forest)
if 'Random Forest' in resultados:
    rf = resultados['Random Forest']['modelo']
    importancias = pd.Series(rf.feature_importances_, index=X.columns).nlargest(15)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    importancias[::-1].plot(kind='barh', ax=ax2, color='steelblue', alpha=0.85)
    ax2.set_xlabel('Importancia (Gini)')
    ax2.set_title('Top 15 Variables Más Importantes — Random Forest (Clasificación)')
    plt.tight_layout()
    plt.savefig('importancia_variables_clasificacion.png', dpi=150, bbox_inches='tight')
    print("Gráfico guardado: importancia_variables_clasificacion.png")
    plt.show()
