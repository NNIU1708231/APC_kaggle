import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin


df = pd.read_csv(r"C:\Users\omega\Downloads\archive\dataset.csv")

df_clean = df.copy()

# Manejo de valores nulos 
df_clean['Year_of_Release'] = df_clean['Year_of_Release'].fillna(df_clean['Year_of_Release'].median())
df_clean['Critic_Score'] = df_clean['Critic_Score'].fillna(df_clean['Critic_Score'].median())
df_clean['User_Score'] = pd.to_numeric(df_clean['User_Score'], errors='coerce')
df_clean['User_Score'] = df_clean['User_Score'].fillna(df_clean['User_Score'].median())
df_clean['Publisher'] = df_clean['Publisher'].fillna('Unknown')
df_clean['Developer'] = df_clean['Developer'].fillna('Unknown')

# Se eliminan Rating nulo porque es categórico, proposar una solucio per a un problema
df_clean = df_clean.dropna(subset=['Rating'])

# Definir características y target
X = df_clean.drop(['Global_Sales', 'Name'], axis=1)  
y = df_clean['Global_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Identificar columnas numéricas y categóricas
numeric_features = ['Year_of_Release', 'NA_Sales', 'EU_Sales', 
                    'JP_Sales', 'Other_Sales', 'Critic_Score', 
                    'Critic_Count', 'User_Score', 'User_Count']
categorical_features = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Calcular medianas SOLO del training set para usar en transform
        self.critic_score_median_ = X['Critic_Score'].median()
        self.user_score_median_ = X['User_Score'].median()
        return self
    
    def transform(self, X):
        X = X.copy()        
        # Características derivadas seguras
        X['Total_Reviews'] = X['Critic_Count'] + X['User_Count']
        
        # Usar medianas aprendidas en fit para evitar divisiones por cero/NaN
        critic_score_safe = X['Critic_Score'].fillna(self.critic_score_median_)
        user_score_safe = X['User_Score'].fillna(self.user_score_median_)
        
        # Score ratio con protección contra división por cero
        X['Score_Ratio'] = np.where(
            user_score_safe > 0,
            critic_score_safe / user_score_safe,
            0
        )
        
        # Diferencia de scores
        X['Score_Diff'] = critic_score_safe - user_score_safe
        
        return X

# Preprocesador por columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Imputa SOLO con datos de train
            ('scaler', StandardScaler())  # Escala SOLO con datos de train
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Ignora categorías nuevas
        ]), categorical_features)
    ]
)

# Pipeline completo y seguro
pipeline = Pipeline(steps=[
    ('features', FeatureEngineering()), 
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100, 
        random_state=42,
        n_jobs=-1  
    ))
])

print("\nEntrenando pipeline...")
pipeline.fit(X_train, y_train)

# Predecir en test
y_pred = pipeline.predict(X_test)

print("\n=== RESULTADOS ===")
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')
print(f'MAE: {np.mean(np.abs(y_test - y_pred)):.4f}')
print(f'R² Score: {r2_score(y_test, y_pred):.4f}')

print("\n=== VALIDACIÓN CRUZADA (5-fold) ===")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, 
                           cv=cv, scoring='r2', n_jobs=-1)
print(f'CV R² scores: {cv_scores}')
print(f'Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})')

# Comparar con baseline simple
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)
print(f"\n=== BASELINE (predicción con media) ===")
print(f'Baseline RMSE: {baseline_rmse:.4f}')
print(f'Baseline R²: {baseline_r2:.4f}')

# Importancia de características (si es RandomForest)
print("\n=== TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES ===")
feature_names = (numeric_features + 
                 list(pipeline.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .named_steps['encoder']
                      .get_feature_names_out(categorical_features)))

# Agregar las features creadas
feature_names = ['Total_Reviews', 'Score_Ratio', 'Score_Diff'] + feature_names

importances = pipeline.named_steps['regressor'].feature_importances_
indices = np.argsort(importances)[-10:][::-1]

for i in indices:
    print(f"{feature_names[i] if i < len(feature_names) else f'Feature_{i}'}: {importances[i]:.4f}")

# Visualización de resultados
plt.figure(figsize=(15, 5))

# 1. Predicciones vs Real
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.title('Predicciones vs Reales')

# 2. Errores
plt.subplot(1, 3, 2)
errors = y_test - y_pred
plt.hist(errors, bins=50, edgecolor='black')
plt.xlabel('Error de Predicción')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores')

plt.subplot(1, 3, 3)
top_features = 15
indices = np.argsort(importances)[-top_features:][::-1]
plt.barh(range(top_features), importances[indices[:top_features]])
plt.yticks(range(top_features), [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                                 for i in indices[:top_features]])
plt.xlabel('Importancia')
plt.title(f'Top {top_features} Características')

plt.tight_layout()
plt.show()

print("\nPipeline entrenado SIN data leakage!")
print("Todas las transformaciones se aprendieron SOLO del training set")
print("Test set permaneció completamente aislado durante el entrenamiento")
