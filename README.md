# Trabajo Practico Metodologia del Aprendizaje

En este trabajo utilizamos una red neuronal construida con Keras (Python) para aproximar faltas en partidos entre 2 equipos conocidos.

Para entrenar la red (los archivos entrenados están subidos) correr el comando:
`python3 train_validate_test.py 1000`
1000 siendo los epochs deseados.
Al terminar el script le preguntara si quiere graficar y guardar los resultados.

Para probar la red con varios partidos aleatorios para ver que tan disperso son las predicciones (que no solo predice un promedio) correr:
`python3 graph_random_inputs.py 500`
500 siendo la cantidad de partidos aleatorios.