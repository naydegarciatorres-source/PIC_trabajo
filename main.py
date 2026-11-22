
import cv2
import numpy as np
import time
import argparse

from reconocimiento_voz_IA import TextToSpeech
from reconocimiento_voz_usuario import escuchar_usuario


ANCHO, ALTO = 640, 480
TIEMPO = 60
RONDAS = 5


def mostrar_info(frame, ronda, puntos_jugador, puntos_ia, tiempo):

    cv2.putText(frame, "PICTIONARY", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Ronda: {ronda}/{RONDAS}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Tu: {puntos_jugador} | IA: {puntos_ia}",
                (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    cv2.putText(frame, f"Tiempo: {int(tiempo)}", (350, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def run_game(num_rounds=RONDAS, round_time=TIEMPO):

    tts = TextToSpeech()

    camara = cv2.VideoCapture(0)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)

    if not camara.isOpened():
        print("Error al abrir la cámara")
        return

    canvas = np.zeros((ALTO, ANCHO), dtype=np.uint8)

    puntos_jugador = 0
    puntos_ia = 0

    try:

        tts.speak_sync("¡Bienvenido al Pictionary!")

        for ronda in range(1, num_rounds + 1):

            print(f"Ronda {ronda}")
            tts.nueva_ronda(ronda, num_rounds)

            canvas[:] = 0
            inicio = time.time()
            terminado = False

            # Bucle principal de la ronda
            while not terminado:

                ret, frame = camara.read()

                if not ret:
                    print("No se ha podido recibir imagen de la cámara")
                    break

                frame = cv2.flip(frame, 1)

                tiempo = round_time - (time.time() - inicio)

                if tiempo <= 0:
                    tts.tiempo_agotado()
                    break

                frame = mostrar_info(
                    frame, ronda, puntos_jugador, puntos_ia, tiempo
                )

                cv2.imshow("Pictionary", frame)
                cv2.imshow("Lienzo", canvas)

                tecla = cv2.waitKey(1) & 0xFF

                if tecla == ord(' '):
                    terminado = True

                elif tecla == ord('c'):
                    canvas[:] = 0

                elif tecla == ord('q'):
                    return

            # Comprobar si hay dibujo
            if np.sum(canvas) < 100:
                print("No hay dibujo.")

            # Predicción temporal de la IA
            prediccion = "gato"
            confianza = 0.85

            print("La IA ha terminado de analizar el dibujo.")

            # Pregunta breve para el jugador
            tts.speak_sync("¿Qué has dibujado?")

            # Dar paso al reconocimiento de voz
            print("\n¡TU TURNO! Habla ahora.")
            respuesta = escuchar_usuario()

            print("Respuesta del jugador:", respuesta)

            # Comparar respuesta con la predicción
            if respuesta and prediccion.lower() in respuesta.lower():

                puntos_ia += 1
                tts.speak_sync("¡Correcto!")

            elif respuesta:

                puntos_jugador += 1
                tts.speak_sync("No has acertado. Era gato.")

            else:

                tts.speak_sync("No he podido entenderte.")

        # Resultado final
        tts.fin_juego(puntos_jugador, puntos_ia)

        print("\nFIN DEL JUEGO")
        print(f"Jugador: {puntos_jugador}")
        print(f"IA: {puntos_ia}")

    finally:

        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--rounds", "-r", type=int, default=RONDAS)
    parser.add_argument("--time", "-t", type=int, default=TIEMPO)

    args = parser.parse_args()

    run_game(args.rounds, args.time)
