
import threading
import pyttsx3


class TextToSpeech:

    def __init__(self):
        self.rate = 150
        self.volume = 1.0
        self.hablando = False

    def speak(self, texto):

        def reproducir():
            self.hablando = True

            try:
                voz = pyttsx3.init()
                voz.setProperty('rate', self.rate)
                voz.setProperty('volume', self.volume)

                # Buscar voz en español
                for voice in voz.getProperty('voices'):
                    if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
                        voz.setProperty('voice', voice.id)
                        break

                voz.say(texto)
                voz.runAndWait()
                voz.stop()

            except Exception as e:
                print("Error de voz:", e)

            self.hablando = False

        threading.Thread(target=reproducir, daemon=True).start()

    def speak_sync(self, texto):
        voz = pyttsx3.init()
        voz.say(texto)
        voz.runAndWait()

    def is_speaking(self):
        return self.hablando

    def bienvenida(self):
        self.speak("¡Bienvenido al Pictionary!")

    def instrucciones(self):
        self.speak("Levanta el dedo para dibujar y cierra el puño para borrar.")

    def nueva_ronda(self, ronda, total):
        self.speak(f"Ronda {ronda} de {total}. ¡Prepárate!")

    def anunciar_prediccion(self, prediccion, confianza):
        porcentaje = int(confianza * 100)
        self.speak(f"Creo que es un {prediccion}, con un {porcentaje} por ciento de confianza.")

    def resultado_correcto(self):
        self.speak("¡He acertado! Punto para la máquina.")

    def resultado_incorrecto(self, respuesta):
        self.speak(f"Me he equivocado. Era un {respuesta}. Punto para ti.")

    def fin_juego(self, jugador, ia):
        self.speak(f"Fin del juego. Tú tienes {jugador} puntos y yo {ia}.")

    def tiempo_agotado(self):
        self.speak("¡Se acabó el tiempo!")
