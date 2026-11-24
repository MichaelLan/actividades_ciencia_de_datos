#import "presentacion.typ": *
= Trabajo futuro
- Validar con particiones temporales (entrenar con 2015–2016 y evaluar en 2017) para estimar el desempeño en reservas futuras.
- Reconstruir los predictores al momento de crear la reserva, en colaboración con los hoteles, para convertir el clasificador en un sistema de alerta temprana.
- Elegir el umbral de decisión con una matriz de costes de sobreventa y de habitaciones vacías, y calibrar las probabilidades.
- Complementar la importancia por permutación con valores SHAP y dependencias parciales para describir la dirección de los efectos.
- Ampliar la búsqueda de hiperparámetros más allá de tres configuraciones por modelo y guardar el ganador (`best_params_`) junto con los artefactos.
