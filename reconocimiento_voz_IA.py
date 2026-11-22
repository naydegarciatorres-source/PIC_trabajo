import threading
import pyttsx3

class TextToSpeech:
    # voz del juego
    def __init__(self):
        self.rate = 160
        self.volume = 1.0
        self.hablando = False
        self._lock = threading.Lock()

    def _configurar_voz(self, voz):
        voz.setProperty('rate', self.rate)
        voz.setProperty('volume', self.volume)
        try:
            for voice in voz.getProperty('voices'):
                nombre = voice.name.lower()
                idiomas = [str(lang).lower() for lang in getattr(voice, 'languages', [])]
                if 'spanish' in nombre or 'español' in nombre or any('es' in l for l in idiomas):
                    voz.setProperty('voice', voice.id)
                    break
        except Exception:
            pass

    def speak_sync(self, texto):
        # para que no hable encima del usuario
        with self._lock:
            self.hablando = True
            try:
                voz = pyttsx3.init()
                self._configurar_voz(voz)
                voz.say(texto)
                voz.runAndWait()
                voz.stop()
            except Exception as e:
                print("Error de voz:", e)
            finally:
                self.hablando = False

    def speak(self, texto):
        self.speak_sync(texto)

    def is_speaking(self):
        return self.hablando

    def bienvenida(self):
        self.speak_sync("¡Bienvenido al Pictionary!")

    def instrucciones(self):
        self.speak_sync("Levanta el dedo para dibujar y cierra el puño para borrar.")

    def nueva_ronda(self, ronda, total):
        self.speak_sync(f"Ronda {ronda} de {total}. ¡Prepárate!")

    def anunciar_prediccion(self, prediccion, confianza):
        porcentaje = int(confianza * 100)
        self.speak_sync(f"Creo que has dibujado: {prediccion}, con un {porcentaje} por ciento de confianza.")

    def resultado_correcto(self):
        self.speak_sync("¡He acertado! Has dibujado genial, te llevas el punto.")

    def resultado_incorrecto(self, respuesta):
        self.speak_sync(f"¡Vaya, no lo he adivinado! Era {respuesta}, así que el punto me lo llevo yo.")

    def fin_juego(self, jugador, ia):
        if ia > jugador:
            self.speak_sync(f"¡Fin del juego! Toma, te he ganado {ia} a {jugador}.")
        elif jugador > ia:
            self.speak_sync(f"¡Fin del juego! Vaya, me has ganado {jugador} a {ia}, muy bien jugado.")
        else:
            self.speak_sync(f"¡Fin del juego! Hemos empatado a {jugador}, qué partida más reñida.")

    def tiempo_agotado(self):
        self.speak_sync("¡Se acabó el tiempo!")
