import cv2
import numpy as np
import time
import argparse
import random

from reconocimiento_voz_IA import TextToSpeech
from reconocimiento_voz_usuario import escuchar_usuario
import beit_classifier as beit
from hand_detection import capturar_dibujo
from traducciones import traducir

ANCHO, ALTO = 640, 480
TIEMPO = 30
RONDAS = 3

def run_game(num_rounds=RONDAS, round_time=TIEMPO):
    tts = TextToSpeech()
    clf = beit.get_classifier()
    palabras_disponibles = clf.classes

    puntos_jugador = 0
    puntos_ia = 0

    try:
        tts.bienvenida()
        tts.instrucciones()

        for ronda in range(1, num_rounds + 1):
            print(f"\n==================== RONDA {ronda}/{num_rounds} ====================")
            tts.nueva_ronda(ronda, num_rounds)

            target_en = random.choice(palabras_disponibles)
            target_es = traducir(target_en)

            print(f"Palabra a dibujar: {target_es.upper()} ({target_en})")
            tts.speak_sync(f"Tu palabra a dibujar es: {target_es}.")

            marcador_texto = f"Tu: {puntos_jugador} | IA: {puntos_ia}"

            canvas, trazos = capturar_dibujo(
                devolver_trazos=True,
                palabra=target_es,
                marcador=marcador_texto,
                tiempo_max=round_time
            )

            if canvas is None:
                print("Ronda cancelada por el usuario.")
                continue

            # clasificar con beit
            if trazos:
                resultados = beit.predict_strokes(trazos)
            else:
                resultados = beit.predict(canvas)

            prediccion_en, confianza = resultados[0] if resultados else (None, 0.0)
            prediccion_es = traducir(prediccion_en) if prediccion_en else "nada"

            print(f"Prediccion IA: {prediccion_es} ({prediccion_en}) con {confianza*100:.1f}% de confianza")

            if prediccion_en is not None:
                tts.anunciar_prediccion(prediccion_es, confianza)
            else:
                tts.speak_sync("No he podido identificar ningun trazo.")

            time.sleep(1.0)
            tts.speak_sync("Dime, he acertado?")
            time.sleep(0.5)

            print("\n¡TU TURNO! Habla ahora por el microfono...")
            respuesta_voz = escuchar_usuario()
            print(f"Respuesta reconocida: '{respuesta_voz}'")

            acierto_ia = False
            resp = respuesta_voz.lower().strip()

            if prediccion_en and prediccion_en.lower() == target_en.lower():
                acierto_ia = True
            elif any(afirmativo in resp for afirmativo in ["si", "sí", "correcto", "exacto", "acertaste", "has acertado"]):
                acierto_ia = True
            elif prediccion_es in resp or (prediccion_en and prediccion_en.lower() in resp):
                acierto_ia = True

            # si acierto, punto para ti; si fallo, me llevo el punto yo
            if acierto_ia:
                puntos_jugador += 1
                tts.resultado_correcto()
                print(f"¡He acertado! Punto para ti. (Era: {target_es})")
            else:
                puntos_ia += 1
                tts.resultado_incorrecto(target_es)
                print(f"¡No lo he adivinado! Punto para mí. (Era: {target_es})")

            print(f"Marcador actual -> Tu: {puntos_jugador} | IA: {puntos_ia}")

        tts.fin_juego(puntos_jugador, puntos_ia)
        print("\n==================== FIN DEL JUEGO ====================")
        print(f"Resultado final: Tu {puntos_jugador} - {puntos_ia} IA")
        if puntos_ia > puntos_jugador:
            print("¡Te he ganado!")
        elif puntos_jugador > puntos_ia:
            print("¡Me has ganado, bien jugado!")
        else:
            print("¡Hemos empatado!")

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pictionary con Inteligencia Artificial")
    parser.add_argument("--rounds", "-r", type=int, default=RONDAS, help="Numero de rondas")
    parser.add_argument("--time", "-t", type=int, default=TIEMPO, help="Tiempo por ronda")
    args = parser.parse_args()
    run_game(args.rounds, args.time)
