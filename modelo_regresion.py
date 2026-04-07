"""
MODELO DE REGRESIÓN — Ames Housing Dataset
Objetivo: Predecir el precio de venta (SalePrice) de viviendas

Requisitos:
    pip install pandas scikit-learn matplotlib seaborn numpy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────
df = pd.read_csv('ames_housing_processed.csv')

X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Variable objetivo: SalePrice  |  Min: ${y.min():,.0f}  Max: ${y.max():,.0f}  Media: ${y.mean():,.0f}")

# ─────────────────────────────────────────────────────────
# 2. DIVISIÓN DE DATOS — 75% entrenamiento / 25% prueba
# ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"\nEntrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba:        {X_test.shape[0]} muestras")

# ─────────────────────────────────────────────────────────
# 3. MODELOS CANDIDATOS
# ─────────────────────────────────────────────────────────
modelos = {
    'Ridge Regression':        Ridge(alpha=10.0),
    'Decision Tree (max_d=5)': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest':           RandomForestRegressor(
                                   n_estimators=200,
                                   max_depth=12,
                                   min_samples_leaf=3,
                                   random_state=42,
                                   n_jobs=-1),
}

resultados = {}
print("\n" + "="*55)
print(f"{'Modelo':<30} {'RMSE':>9} {'MAE':>9} {'R²':>8} {'CV R²':>8}")
print("="*55)

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    cv   = cross_val_score(modelo, X_train, y_train, cv=5, scoring='r2').mean()
    resultados[nombre] = dict(modelo=modelo, pred=pred, rmse=rmse, mae=mae, r2=r2, cv_r2=cv)
    print(f"{nombre:<30} ${rmse:>8,.0f} ${mae:>8,.0f} {r2:>8.4f} {cv:>8.4f}")

print("="*55)

# ─────────────────────────────────────────────────────────
# 4. MEJOR MODELO
# ─────────────────────────────────────────────────────────
mejor_nombre = max(resultados, key=lambda k: resultados[k]['r2'])
mejor = resultados[mejor_nombre]
print(f"\n★ Mejor modelo: {mejor_nombre}")
print(f"  R²   = {mejor['r2']:.4f}  (explica el {mejor['r2']*100:.1f}% de la varianza)")
print(f"  RMSE = ${mejor['rmse']:,.0f}")
print(f"  MAE  = ${mejor['mae']:,.0f}")
print(f"  CV R²= {mejor['cv_r2']:.4f}  (sin overfitting si ≈ R² test)")

# ─────────────────────────────────────────────────────────
# 5. VISUALIZACIONES
# ─────────────────────────────────────────────────────────
pred_best = mejor['pred']
residuals = y_test - pred_best

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(f'Evaluación del Modelo de Regresión\n{mejor_nombre} — Ames Housing',
             fontsize=14, fontweight='bold')

# 5.1 Real vs Predicho
ax = axes[0, 0]
mn, mx = min(y_test.min(), pred_best.min()), max(y_test.max(), pred_best.max())
ax.scatter(y_test, pred_best, alpha=0.5, color='steelblue', s=25)
ax.plot([mn, mx], [mn, mx], '--r', lw=2, label='Predicción perfecta')
ax.set_xlabel('Precio Real ($)'); ax.set_ylabel('Precio Predicho ($)')
ax.set_title('Real vs. Predicho')
ax.legend()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

# 5.2 Residuos vs Predicho
ax = axes[0, 1]
ax.scatter(pred_best, residuals, alpha=0.5, color='seagreen', s=25)
ax.axhline(0, color='red', lw=2, ls='--')
ax.set_xlabel('Precio Predicho ($)'); ax.set_ylabel('Residuo ($)')
ax.set_title('Gráfico de Residuos')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

# 5.3 Distribución de Residuos
ax = axes[1, 0]
ax.hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', lw=2, ls='--')
ax.set_xlabel('Residuo ($)'); ax.set_ylabel('Frecuencia')
ax.set_title('Distribución de Residuos')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

# 5.4 Comparativa R² Test vs CV
ax = axes[1, 1]
nombres_cortos = ['Ridge', 'Dec. Tree', 'Random\nForest']
r2_test = [resultados[n]['r2']   for n in resultados]
r2_cv   = [resultados[n]['cv_r2'] for n in resultados]
x = np.arange(3); w = 0.35
ax.bar(x - w/2, r2_test, w, label='R² Test',     color='steelblue', alpha=0.9)
ax.bar(x + w/2, r2_cv,   w, label='R² CV (Train)', color='seagreen', alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(nombres_cortos)
ax.set_ylabel('R²'); ax.set_title('R² Test vs CV — Comparación de Modelos')
ax.set_ylim(0, 1.05); ax.legend()
for i, (t, c) in enumerate(zip(r2_test, r2_cv)):
    ax.text(i - w/2, t + 0.01, f'{t:.3f}', ha='center', va='bottom', fontsize=8)
    ax.text(i + w/2, c + 0.01, f'{c:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('resultados_regresion.png', dpi=150, bbox_inches='tight')
print("\nGráfico guardado: resultados_regresion.png")
plt.show()

# Importancia de variables (Random Forest)
if 'Random Forest' in resultados:
    rf = resultados['Random Forest']['modelo']
    importancias = pd.Series(rf.feature_importances_, index=X.columns).nlargest(20)
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    importancias[::-1].plot(kind='barh', ax=ax2, color='steelblue', alpha=0.85)
    ax2.set_xlabel('Importancia (Gini)')
    ax2.set_title('Top 20 Variables Más Importantes — Random Forest (Regresión)')
    plt.tight_layout()
    plt.savefig('importancia_variables_regresion.png', dpi=150, bbox_inches='tight')
    print("Gráfico guardado: importancia_variables_regresion.png")
    plt.show()
