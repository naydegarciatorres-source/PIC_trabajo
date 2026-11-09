# PICtionary
## RESUMEN
La idea del proyecto es desarrollar una versión del Pictionary mediante técnicas de visión por computador.

El usuario dibujara en el aire con su mano el dibujo que quiera, que será detectado con una cámara.

Posteriormente la red neuronal se encargará de analizar y clasificar el dibujo. Tras tomar una decisión, el usario dirá en voz alta el dibujo que había hecho y se comprobará con la respuesta de la red neuronal.

## DETECCIÓN DE MANOS
Se usará MediaPipe Hands para poder distinguir las siguientes situaciones:
- Dedo índice levantado: Pintar
- Puño cerrado: Borrar
- Mano abierta: Pausar

Se desarrollará un algoritmo de suavizado de trazado para evitar temblores.

Por último, como sálida devuelve el dibujo en un lienzo en blanco y negro (Matriz NumPy de OpenCV).
## RED NEURONAL
Se cargará el modelo de la red neuronal mediante PyTorch, TensorFlow o Hugging Face, entrenado en Quick,Draw!

Luego, se hará el preprocesamiento de la imagen (redimensionar, invertir colores, normalizar,... lo que se vea necesario).

Por último, predice la respuesta y la devuelve con su porcentaje de confianza.
## DESARROLLO DEL JUEGO
Será el búcle que se encargará de implementar los otros dos modulos y se gestionará la parte del control de voz:
- Módulo TTS con pyttsx3 
- Módulo STT con SpeechRecognition 