import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\nFASE 1: ANÁLISIS EXPLORATORIO DE DATOS")
print("-"*50)

df = pd.read_csv(r"C:\Users\omega\Downloads\archive\dataset.csv")
print(f"Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")

df_analysis = df.copy()

# Convertir User_Score a numérico
df_analysis['User_Score'] = pd.to_numeric(df_analysis['User_Score'], errors='coerce')

# Imputar valores nulos para análisis (no para modelo)
df_analysis['Critic_Score'] = df_analysis['Critic_Score'].fillna(df_analysis['Critic_Score'].median())
df_analysis['User_Score'] = df_analysis['User_Score'].fillna(df_analysis['User_Score'].median())
df_analysis['Year_of_Release'] = df_analysis['Year_of_Release'].fillna(df_analysis['Year_of_Release'].median())

# Crear características derivadas para análisis
df_analysis['Total_Sales_Calculated'] = df_analysis[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum(axis=1)
df_analysis['Score_Diff'] = df_analysis['Critic_Score'] - df_analysis['User_Score'] * 10
df_analysis['Review_Count_Total'] = df_analysis['Critic_Count'] + df_analysis['User_Count']

print("\nCalculando correlaciones entre variables...")

numeric_cols = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',
                'Critic_Score', 'User_Score', 'Critic_Count', 'User_Count',
                'Year_of_Release', 'Total_Sales_Calculated', 'Score_Diff', 'Review_Count_Total']

df_numeric = df_analysis[numeric_cols].dropna()
correlation_matrix = df_numeric.corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8})

plt.title('MATRIZ DE CORRELACIÓN ENTRE VARIABLES NUMÉRICAS', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nVisualizando relaciones más importantes...")

threshold = 0.3
strong_correlations = []

for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > threshold:
            strong_correlations.append({
                'Variable1': numeric_cols[i],
                'Variable2': numeric_cols[j],
                'Correlación': corr,
                'Abs_Corr': abs(corr)
            })

corr_df = pd.DataFrame(strong_correlations).sort_values('Abs_Corr', ascending=False)

plt.figure(figsize=(12, 8))
top_correlations = corr_df.head(15).copy()
top_correlations['Label'] = top_correlations.apply(
    lambda x: f"{x['Variable1'][:15]} ↔ {x['Variable2'][:15]}", axis=1)

bars = plt.barh(range(len(top_correlations)), top_correlations['Correlación'].values)
plt.yticks(range(len(top_correlations)), top_correlations['Label'].values)

for i, bar in enumerate(bars):
    if top_correlations.iloc[i]['Correlación'] > 0:
        bar.set_color('red')
    else:
        bar.set_color('blue')

plt.xlabel('Valor de Correlación')
plt.title('🔝 TOP 15 CORRELACIONES MÁS FUERTES', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\nAnalizando variables categóricas...")

categorical_vars = ['Platform', 'Genre', 'Rating']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, cat_var in enumerate(categorical_vars):
    ax = axes[idx]
    
    top_categories = df_analysis[cat_var].value_counts().head(8).index
    data_filtered = df_analysis[df_analysis[cat_var].isin(top_categories)]
    
    box_data = [data_filtered[data_filtered[cat_var] == cat]['Global_Sales'].values 
                for cat in top_categories]
    
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_categories)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xticklabels([str(cat)[:12] for cat in top_categories], rotation=45, ha='right')
    ax.set_ylabel('Global Sales (millones)')
    ax.set_title(f'Ventas por {cat_var}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('DISTRIBUCIÓN DE VENTAS POR CATEGORÍAS', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("INSIGHTS DEL ANÁLISIS EXPLORATORIO")
print("="*60)

print("\nTOP 5 CORRELACIONES CON GLOBAL_SALES:")
top_positive = correlation_matrix['Global_Sales'].sort_values(ascending=False).head(6)
for i, (var, corr) in enumerate(top_positive.items()):
    if var != 'Global_Sales':
        strength = "FUERTE" if abs(corr) > 0.7 else "MODERADA" if abs(corr) > 0.3 else "DÉBIL"
        print(f"  {i}. {var:25} : {corr:.4f} ({strength})")

print("\nTOP PLATAFORMAS POR VENTAS:")
platform_stats = df_analysis.groupby('Platform')['Global_Sales'].agg(['median', 'count']).sort_values('median', ascending=False).head(3)
for platform, row in platform_stats.iterrows():
    print(f"  • {platform:20} : {row['median']:.2f}M (n={row['count']})")

print("\nTOP GÉNEROS POR VENTAS:")
genre_stats = df_analysis.groupby('Genre')['Global_Sales'].agg(['median', 'count']).sort_values('median', ascending=False).head(3)
for genre, row in genre_stats.iterrows():
    print(f"  • {genre:20} : {row['median']:.2f}M (n={row['count']})")


print("\n" + "="*70)
print("FASE 2: ENTRENAMIENTO DE MODELO PREDICTIVO")
print("="*70)

print("\nPreparando datos para modelo (evitando data leakage)...")

df_model = df.copy()
df_model['User_Score'] = pd.to_numeric(df_model['User_Score'], errors='coerce')
df_model = df_model.dropna(subset=['Rating'])

X = df_model.drop(['Global_Sales', 'Name'], axis=1)
y = df_model['Global_Sales']

# DIVIDIR PRIMERO 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Dataset dividido correctamente:")
print(f"   • Training set: {len(X_train)} muestras")
print(f"   • Test set: {len(X_test)} muestras")
print(f"   • Características iniciales: {X_train.shape[1]}")

numeric_features = ['Year_of_Release', 'NA_Sales', 'EU_Sales', 
                    'JP_Sales', 'Other_Sales', 'Critic_Score', 
                    'Critic_Count', 'User_Score', 'User_Count']

categorical_features = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']

print(f"\n📋 Variables identificadas:")
print(f"   • Numéricas: {len(numeric_features)}")
print(f"   • Categóricas: {len(categorical_features)}")

# 3. FEATURE ENGINEERING SEGURO (dentro del pipeline)
class SafeFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_ = list(X.columns)
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Total de reseñas
        if 'Critic_Count' in X.columns and 'User_Count' in X.columns:
            X['Total_Reviews'] = X['Critic_Count'] + X['User_Count']
        
        # Ventas calculadas
        sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        if all(col in X.columns for col in sales_cols):
            X['Calculated_Total_Sales'] = X[sales_cols].sum(axis=1)
        
        # Proporciones regionales
        if 'NA_Sales' in X.columns and 'Calculated_Total_Sales' in X.columns:
            X['NA_Sales_Ratio'] = X['NA_Sales'] / (X['Calculated_Total_Sales'] + 1e-10)
            X['EU_Sales_Ratio'] = X['EU_Sales'] / (X['Calculated_Total_Sales'] + 1e-10)
            X['JP_Sales_Ratio'] = X['JP_Sales'] / (X['Calculated_Total_Sales'] + 1e-10)
        
        # Características temporales
        if 'Year_of_Release' in X.columns:
            X['Years_Since_Release'] = 2024 - X['Year_of_Release']
            X['Decade'] = (X['Year_of_Release'] // 10) * 10
        
        return X

# 4. PIPELINE DE PREPROCESAMIENTO
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  
            ('scaler', StandardScaler())  
        ]), numeric_features),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                min_frequency=0.01
            ))
        ]), categorical_features)
    ],
    remainder='drop'
)

# 5. PIPELINE COMPLETO
pipeline = Pipeline([
    ('features', SafeFeatureEngineering()),
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    ))
])


print("\n🚀 Entrenando modelo 1")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n" + "="*50)
print("RESULTADOS DEL MODELO 1")
print("="*50)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

print(f"\nMÉTRICAS DE REGRESIÓN:")
print(f"   • R² Score: {r2:.4f} (Varianza explicada)")
print(f"   • RMSE: {rmse:.4f} (Error cuadrático medio)")
print(f"   • MAE: {mae:.4f} (Error absoluto medio)")
print(f"   • MAPE: {mape:.2f}% (Error porcentual medio)")

print("\n" + "="*70)
print("ENTRENANDO MODELO 2 (SIN VENTAS REGIONALES)")
print("="*70)

numeric_features_2 = ['Year_of_Release', 'Critic_Score',
                      'Critic_Count', 'User_Score', 'User_Count']

categorical_features_2 = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']

X2 = df_model.drop(['Global_Sales', 'Name',
                    'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1)
y2 = df_model['Global_Sales']

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, shuffle=True
)
preprocessor_2 = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features_2),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features_2)
    ]
)

pipeline_2 = Pipeline([
    ('features', SafeFeatureEngineering()),   
    ('preprocessor', preprocessor_2),
    ('regressor', RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nEntrenando modelo 2...")
pipeline_2.fit(X2_train, y2_train)

y2_pred = pipeline_2.predict(X2_test)

rmse2 = np.sqrt(mean_squared_error(y2_test, y2_pred))
mae2 = mean_absolute_error(y2_test, y2_pred)
r2_2 = r2_score(y2_test, y2_pred)
mape2 = np.mean(np.abs((y2_test - y2_pred) / (y2_test + 1e-10))) * 100

print("\nRESULTADOS MODELO 2 (SIN ventas regionales):")
print(f"   • R² Score: {r2_2:.4f}")
print(f"   • RMSE: {rmse2:.4f}")
print(f"   • MAE: {mae2:.4f}")
print(f"   • MAPE: {mape2:.2f}%")

cv_2 = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_2 = cross_val_score(pipeline_2, X2_train, y2_train,
                              cv=cv_2, scoring='r2', n_jobs=-1)

print("\nVALIDACIÓN CRUZADA MODELO 2:")
print(f"   • R² promedio: {cv_scores_2.mean():.4f} ± {cv_scores_2.std()*2:.4f}")

print("\n" + "="*70)
print("COMPARACIÓN ENTRE MODELOS")
print("="*70)
print(f"Modelo 1 (con ventas regionales): R² = {r2:.4f}")
print(f"Modelo 2 (sin ventas regionales): R² = {r2_2:.4f}")
print(f"Diferencia: {r2 - r2_2:.4f} puntos de R²")

print(f"\n🔄 VALIDACIÓN CRUZADA (5-fold):")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, 
                           cv=cv, scoring='r2', n_jobs=-1)

print(f"   • R² por fold: {np.round(cv_scores, 4)}")
print(f"   • R² promedio: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")
print(f"   • Rango: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")

baseline_pred = np.full_like(y_test, y_train.median())
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)

print(f"\n COMPARACIÓN CON BASELINE (predecir mediana):")
print(f"   • Baseline R²: {baseline_r2:.4f}")
print(f"   • Mejora en R²: {r2 - baseline_r2:.4f} ({((r2 - baseline_r2)/abs(baseline_r2 + 1e-10))*100:.1f}% mejor)")
print(f"   • Mejora en RMSE: {((baseline_rmse - rmse)/baseline_rmse)*100:.1f}%")

print(f"\n TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")

cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
cat_features = cat_encoder.get_feature_names_out(categorical_features)
    
all_features = list(numeric_features) + list(cat_features)
    
extra_features = ['Total_Reviews', 'Calculated_Total_Sales', 
                     'NA_Sales_Ratio', 'EU_Sales_Ratio', 'JP_Sales_Ratio',
                     'Years_Since_Release', 'Decade']
all_features = extra_features + all_features
    
importances = pipeline.named_steps['regressor'].feature_importances_
    
feature_importance_df = pd.DataFrame({
        'Feature': all_features[:len(importances)],
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
print(feature_importance_df.head(10).to_string(index=False))
    
print(f" No se pudieron obtener nombres de características: {e}")
importances = pipeline.named_steps['regressor'].feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]
for idx in top_indices:
    print(f"   Feature_{idx}: {importances[idx]:.4f}")

print(f"\nANÁLISIS DE ERRORES:")
print(f"   • Error máximo: {np.max(np.abs(y_test - y_pred)):.4f}")
print(f"   • Error mínimo: {np.min(np.abs(y_test - y_pred)):.4f}")
print(f"   • % predicciones con error < 0.5M: {(np.abs(y_test - y_pred) < 0.5).sum() / len(y_test) * 100:.1f}%")
print(f"   • % predicciones con error < 1.0M: {(np.abs(y_test - y_pred) < 1.0).sum() / len(y_test) * 100:.1f}%")

print("\n" + "="*70)
print(" FASE 3: VISUALIZACIONES DE RESULTADOS")
print("="*70)

fig = plt.figure(figsize=(18, 12))

# Predicciones vs Reales
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_pred, alpha=0.6, s=30, edgecolors='w', linewidth=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción perfecta')
ax1.set_xlabel('Ventas Reales (millones)')
ax1.set_ylabel('Ventas Predichas (millones)')
ax1.set_title('Predicciones vs Valores Reales\nR² = {:.3f}'.format(r2))
ax1.legend()
ax1.grid(True, alpha=0.3)

# Distribución de errores
ax2 = plt.subplot(2, 3, 2)
errors = y_test - y_pred
ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7, density=True)
ax2.axvline(x=0, color='r', linestyle='--', label='Error cero')
ax2.axvline(x=errors.mean(), color='g', linestyle='-', label=f'Media: {errors.mean():.3f}')
ax2.set_xlabel('Error de Predicción (Real - Predicho)')
ax2.set_ylabel('Densidad')
ax2.set_title(f'Distribución de Errores\nMAE={mae:.3f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Importancia de características
ax3 = plt.subplot(2, 3, 3)
top_n = 8
if 'feature_importance_df' in locals():
    top_features = feature_importance_df.head(top_n)
    ax3.barh(range(top_n), top_features['Importance'].values)
    ax3.set_yticks(range(top_n))
    ax3.set_yticklabels([f[:20] for f in top_features['Feature'].values])
else:
    top_indices = np.argsort(importances)[-top_n:][::-1]
    ax3.barh(range(top_n), importances[top_indices])
    ax3.set_yticks(range(top_n))
    ax3.set_yticklabels([f'Feature_{i}' for i in top_indices])
ax3.set_xlabel('Importancia')
ax3.set_title(f'Top {top_n} Características\nMás Importantes')
ax3.invert_yaxis()

# Gráfico de burbujas para relaciones clave
ax4 = plt.subplot(2, 3, 4)
sample_data = df_analysis.sample(min(500, len(df_analysis)), random_state=42)
sample_data = sample_data.dropna(subset=['Critic_Score', 'User_Score', 'Global_Sales'])
sample_data = sample_data[sample_data['Global_Sales'] < sample_data['Global_Sales'].quantile(0.95)]

sizes = np.sqrt(sample_data['Critic_Count'].fillna(1)) * 20
scatter = ax4.scatter(sample_data['Critic_Score'], 
                      sample_data['User_Score'] * 10,
                      s=sizes,
                      c=sample_data['Year_of_Release'],
                      cmap='viridis',
                      alpha=0.6)
ax4.set_xlabel('Critic Score')
ax4.set_ylabel('User Score (x10)')
ax4.set_title('Relación Puntuaciones vs Año\n(Tamaño = Reseñas)')
plt.colorbar(scatter, ax=ax4, label='Año')

# Comparación por género (top 5)
ax5 = plt.subplot(2, 3, 5)
top_genres = df_analysis.groupby('Genre')['Global_Sales'].median().sort_values(ascending=False).head(5).index
genre_data = df_analysis[df_analysis['Genre'].isin(top_genres)]
genre_medians = genre_data.groupby('Genre')['Global_Sales'].median().sort_values(ascending=False)

ax5.bar(range(len(genre_medians)), genre_medians.values, color=plt.cm.Set3(range(len(genre_medians))))
ax5.set_xticks(range(len(genre_medians)))
ax5.set_xticklabels(genre_medians.index, rotation=45, ha='right')
ax5.set_ylabel('Ventas Medianas (M)')
ax5.set_title('Top 5 Géneros por Ventas')
ax5.grid(True, alpha=0.3, axis='y')

# Evolución temporal
ax6 = plt.subplot(2, 3, 6)
if len(df_analysis) > 0 and 'Year_of_Release' in df_analysis.columns:
    yearly_avg = df_analysis.groupby('Year_of_Release')['Global_Sales'].mean().dropna()
    ax6.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, color='darkblue')
    ax6.set_xlabel('Año de Lanzamiento')
    ax6.set_ylabel('Ventas Promedio (M)')
    ax6.set_title('Evolución de Ventas por Año')
    ax6.grid(True, alpha=0.3)

plt.suptitle('RESUMEN VISUAL DEL ANÁLISIS COMPLETO', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

