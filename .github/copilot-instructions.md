## Contexto
Proyecto Python para predecir BTTS (Both Teams To Score) en la EFL Cup.
Modelo ML entrenado con historial de partidos, features calculadas solo con
información disponible ANTES del kickoff. 

## Estructura esperada
- train_*.py: entrena el modelo, guarda en models/
- predict_*.py: carga modelo, consulta API de fixtures, escribe CSV de predicciones
- data/: datos históricos (CSV)
- models/: artefactos del modelo 

## Estándares
- Data leakage es el pecado capital: NUNCA usar información posterior al partido
- Nombres de equipos: normalizar a un canónico (ej: "Man Utd" no "Manchester United")
- API: siempre timeout=30, retry con backoff exponencial (máx 3 intentos)
- CSV output: columnas fijas, no cambiar orden ni nombres
- Logging: usar logging module, no print
- Tipos: type hints en todas las funciones públicas
- Seed: random.seed(42) y np.random.seed(42) al inicio de cada script

## Build y test
- Entrenar: python train_btts_efl_cup.py
- Predecir: python predict_btts_efl_cup.py
- Verificar: el CSV de output debe tener filas > 0 y columnas esperadas   
