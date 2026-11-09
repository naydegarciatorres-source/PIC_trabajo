
import speech_recognition as sr


def escuchar_usuario():
    reconocedor = sr.Recognizer()

    try:
        with sr.Microphone() as microfono:
            print("Te estoy escuchando...")
            
            reconocedor.adjust_for_ambient_noise(
                microfono, duration=1
            )

            reconocedor.pause_threshold = 0.8
            reconocedor.dynamic_energy_threshold = True

            print("¡Te escucho! Habla ahora.")

            audio = reconocedor.listen(
                microfono,
                timeout=10,
                phrase_time_limit=8
            )

        print("Reconociendo lo que has dicho...")

        texto = reconocedor.recognize_google(
            audio,
            language="es-ES"
        )

        print("Has dicho:", texto)
        return texto.lower().strip()

    except sr.WaitTimeoutError:
        print("No has hablado a tiempo.")
        return ""

    except sr.UnknownValueError:
        print("No he podido entenderte.")
        return ""

    except sr.RequestError:
        print("Error al conectar con el servicio de reconocimiento.")
        return ""

    except OSError:
        print("No se ha podido acceder al micrófono.")
        return ""
