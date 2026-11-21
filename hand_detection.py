"""
Módulo de dibujo con gestos de la mano.

Uso desde otro archivo:

    from hand_detection import capturar_dibujo

    imagen = capturar_dibujo()            # np.ndarray (alto, ancho) uint8, fondo 255 y trazo 0
    imagen = capturar_dibujo(tamano=(28, 28), recortar=True)   # ya preparada para una red tipo MNIST/QuickDraw
    if imagen is None:
        ...  # el usuario cerró la ventana sin terminar el dibujo

    # También los trazos originales, en píxeles del lienzo (para el clasificador BEiT):
    imagen, trazos = capturar_dibujo(devolver_trazos=True)
    # trazos = [[(x, y), (x, y), ...], [(x, y), ...], ...]  -> (None, []) si se cancela

Gestos:
    - Puño izquierdo cerrado + índice derecho  -> dibujar
    - Puño izquierdo cerrado + palma derecha abierta ~1 s -> borrar todo
    - Pulgar arriba (👍) con la izquierda -> dibujo terminado, se devuelve la imagen
    - 'q' o cerrar una ventana con la X -> cancelar (devuelve None)
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import time
import numpy as np

def dist(a, b):
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))

def check_indice_estirado(mano, umbral=0.80):
    # Si el dedo está recto, la distancia directa nudillo-punta (5→8) es casi igual a la suma de sus tres tramos (5→6 + 6→7 + 7→8)
    recorrido = dist(mano[5], mano[6]) + dist(mano[6], mano[7]) + dist(mano[7], mano[8])
    directa = dist(mano[5], mano[8])
    if directa / recorrido > umbral:
        return True
    else:
        return False

def ratio_dedo(mano, a, b, c, d):
    # Directa / recorrido de un dedo: ~1 = recto, cuanto más bajo, más doblado
    recorrido = dist(mano[a], mano[b]) + dist(mano[b], mano[c]) + dist(mano[c], mano[d])
    return dist(mano[a], mano[d]) / recorrido

def check_puno_cerrado(mano, umbral=0.7):
    # Puño cerrado = índice, corazón, anular y meñique doblados
    # (el pulgar no se mira porque en un puño puede quedar de muchas formas)
    dedos = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    return all(ratio_dedo(mano, *dedo) < umbral for dedo in dedos)

def dedos_doblados(mano_world):
    # Un dedo está doblado si su punta está más cerca de la muñeca (0) que su
    # segunda articulación (PIP). No depende de cómo esté girada la mano.
    dedos = [(6, 8), (10, 12), (14, 16), (18, 20)]  # (PIP, punta) de índice, corazón, anular, meñique
    return all(dist(mano_world[0], mano_world[punta]) < dist(mano_world[0], mano_world[pip])
               for pip, punta in dedos)

def pulgar_arriba(mano_img, margen=0.25):
    # Con coordenadas de imagen (y crece hacia abajo): la punta del pulgar (4)
    # tiene que ser el punto más alto de toda la mano, y sobresalir por encima
    # del resto al menos "margen" veces el tamaño de la palma (0→5)
    tamano_palma = abs(mano_img[0].y - mano_img[5].y) + 1e-6
    resto_mas_alto = min(lm.y for i, lm in enumerate(mano_img) if i not in (3, 4))
    return (resto_mas_alto - mano_img[4].y) / tamano_palma > margen

def check_okay(mano_img, mano_world):
    # 👍 = los cuatro dedos doblados + pulgar apuntando hacia arriba
    return dedos_doblados(mano_world) and pulgar_arriba(mano_img)

def check_palma_abierta(mano_world, umbral=0.85):
    # ✋ = índice, corazón, anular y meñique estirados (casi rectos).
    # Al dibujar solo va estirado el índice, así que no se confunden
    dedos = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    return all(ratio_dedo(mano_world, *dedo) > umbral for dedo in dedos)

class FiltroOneEuro:
    # Suaviza una coordenada. Si el dedo va lento suaviza mucho (quita el temblor)
    # y si va rápido suaviza poco (para que el trazo no vaya con retraso).
    # min_cutoff más bajo = más suave en reposo; beta más alto = menos retraso al moverse rápido
    def __init__(self, min_cutoff=1.0, beta=5.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev, self.dx_prev, self.t_prev = None, 0.0, None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        a_d = self._alpha(self.d_cutoff, dt)
        dx = a_d * (x - self.x_prev) / dt + (1 - a_d) * self.dx_prev
        a = self._alpha(self.min_cutoff + self.beta * abs(dx), dt)
        x_suave = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_suave, dx, t
        return x_suave

def asignar_manos(result):
    # Devuelve (izq, izq_world, der, der_world); None si esa mano no está.
    # Con dos manos se decide por posición: la que está más a la izquierda de la imagen
    # (ya volteada) es la izquierda. Es más fiable que la etiqueta de MediaPipe,
    # que a veces marca las dos manos del mismo lado durante algún frame.
    manos = list(zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness))
    izq = der = (None, None)
    if len(manos) >= 2:
        manos.sort(key=lambda m: m[0][0].x)  # x de la muñeca
        izq, der = manos[0][:2], manos[-1][:2]
    elif len(manos) == 1:
        img, world, lado = manos[0]
        if INVERTIR[lado[0].category_name] == "Right":
            der = (img, world)
        else:
            izq = (img, world)
    return izq[0], izq[1], der[0], der[1]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # pulgar
    (0,5),(5,6),(6,7),(7,8),        # índice
    (5,9),(9,10),(10,11),(11,12),   # corazón
    (9,13),(13,14),(14,15),(15,16), # anular
    (13,17),(0,17),(17,18),(18,19),(19,20)  # meñique
]
INVERTIR = {"Left": "Right", "Right": "Left"}

indice_id = 8  # punta del índice

# Ruta del modelo relativa a este archivo, para que funcione se ejecute desde donde se ejecute
RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

COLOR_TRAZO = (0, 0, 0)  # negro (OpenCV usa BGR)
GROSOR_TRAZO = 5
FRAMES_PARA_BAJAR_LAPIZ = 3   # frames seguidos de puño antes de empezar a pintar (evita rayas por un parpadeo)
FRAMES_TOLERANCIA = 4         # frames que se aguanta sin puño/mano antes de levantar el lápiz
SALTO_MAXIMO = 0.12           # si el dedo "salta" más que esto (fracción de pantalla) en un frame, se empieza trazo nuevo
MIN_PUNTOS_TRAZO = 4          # los trazos más cortos se consideran accidentales y se descartan
FRAMES_RECORTE_GESTO = 4      # puntos que se quitan del final del trazo al hacer el gesto de borrar o el OK
FRAMES_OKAY_NECESARIOS = 10   # el OK tiene que mantenerse unos frames para evitar falsos positivos
FRAMES_BORRAR_NECESARIOS = 30 # ~1 segundo con la palma abierta para borrar

VENTANA_CAMARA = "Hand detection"
VENTANA_LIENZO = "Lienzo"


def a_pixeles(p, w, h):
    return (int(p[0] * w), int(p[1] * h))


class Lienzo:
    # Guarda los trazos (puntos normalizados 0-1) y la imagen pintada a partir de ellos
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.imagen = np.full((h, w, 3), 255, dtype=np.uint8)  # blanco
        self.trazos = []        # trazos terminados; cada uno es una lista de (x, y)
        self.trazo_actual = []

    def reconstruir(self):
        # Repinta la imagen exactamente con los trazos guardados
        self.imagen[:] = 255
        for trazo in self.trazos:
            pts = np.array([a_pixeles(p, self.w, self.h) for p in trazo], dtype=np.int32)
            cv2.polylines(self.imagen, [pts], False, COLOR_TRAZO, GROSOR_TRAZO, cv2.LINE_AA)

    def cerrar_trazo(self):
        # Termina el trazo actual: se guarda si es largo, se descarta si es un toque accidental
        if len(self.trazo_actual) >= MIN_PUNTOS_TRAZO:
            self.trazos.append(self.trazo_actual)
        self.trazo_actual = []
        self.reconstruir()

    def anadir_punto(self, p):
        if self.trazo_actual and math.dist(p, self.trazo_actual[-1]) > SALTO_MAXIMO:
            self.cerrar_trazo()  # salto brusco: trazo nuevo en vez de una línea larga
        if self.trazo_actual:
            cv2.line(self.imagen, a_pixeles(self.trazo_actual[-1], self.w, self.h),
                     a_pixeles(p, self.w, self.h), COLOR_TRAZO, GROSOR_TRAZO, cv2.LINE_AA)
        else:
            cv2.circle(self.imagen, a_pixeles(p, self.w, self.h), GROSOR_TRAZO // 2,
                       COLOR_TRAZO, -1, cv2.LINE_AA)
        self.trazo_actual.append(p)

    def borrar(self):
        self.trazos.clear()
        self.trazo_actual = []
        self.reconstruir()


def preparar_salida(imagen_bgr, tamano=None, recortar=False, escala_grises=True, margen=0.1):
    # Convierte el lienzo en el ndarray que se devuelve
    img = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY) if escala_grises else imagen_bgr.copy()

    if recortar:
        # Recorta al contenido del dibujo, en un cuadrado centrado con un poco de margen,
        # para que el dibujo ocupe toda la imagen aunque se haya hecho pequeño o en una esquina
        gris = img if escala_grises else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gris < 255)
        if len(xs) > 0:
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            lado = int(max(x1 - x0, y1 - y0) * (1 + 2 * margen)) + 1
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            # Se rellena con blanco por si el cuadrado se sale de la imagen
            pad = lado
            relleno = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                         value=255 if escala_grises else (255, 255, 255))
            x_ini, y_ini = cx - lado // 2 + pad, cy - lado // 2 + pad
            img = relleno[y_ini:y_ini + lado, x_ini:x_ini + lado]

    if tamano is not None:
        img = cv2.resize(img, tamano, interpolation=cv2.INTER_AREA)
    return img


def capturar_dibujo(tamano=None, recortar=False, escala_grises=True, camara=0, devolver_trazos=False):
    """
    Abre la cámara para dibujar con gestos y espera a que se haga el 👍 con la mano izquierda.

    Parámetros:
        tamano: (ancho, alto) para redimensionar la imagen devuelta, p. ej. (28, 28).
                None = tamaño de la cámara.
        recortar: True = recorta la imagen al dibujo (cuadrado con margen) antes de redimensionar.
        escala_grises: True = devuelve (alto, ancho); False = devuelve BGR (alto, ancho, 3).
        camara: índice de la cámara para cv2.VideoCapture.
        devolver_trazos: True = devuelve también los trazos (ver abajo).

    Devuelve:
        np.ndarray uint8 con fondo blanco (255) y trazo negro (0),
        o None si se cancela ('q' o cerrar una ventana) antes de terminar el dibujo.
        Con devolver_trazos=True devuelve la tupla (imagen, trazos): trazos es una lista de
        trazos en el orden en que se dibujaron, y cada trazo una lista de puntos (x, y) en
        píxeles del lienzo a tamaño completo (sin recortar ni redimensionar), los mismos con los
        que se pinta la imagen. Si se cancela, devuelve (None, []).
    """
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=RUTA_MODELO),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    cam = cv2.VideoCapture(camara)
    if not cam.isOpened():
        landmarker.close()
        raise RuntimeError(f"No se pudo abrir la cámara {camara}")

    timestamp = 0
    lienzo = None
    filtro_x, filtro_y = FiltroOneEuro(), FiltroOneEuro()
    frames_sin_derecha = 0
    lapiz_bajado = False
    frames_puno = 0
    frames_fallo = 0
    pendientes = []    # puntos de frames dudosos: se pintan si vuelve el puño, se tiran si no
    frames_okay = 0
    frames_borrar = 0
    resultado = None
    trazos_px = []

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Tiempo real en ms (tiene que ir siempre en aumento). Con el tiempo real
            # el tracking de MediaPipe y el suavizado funcionan bien aunque varíen los FPS
            ahora = time.monotonic()
            timestamp = max(timestamp + 1, int(ahora * 1000))

            result = landmarker.detect_for_video(mp_image, timestamp)
            mano_izquierda, mano_izquierda_world, mano_derecha, mano_derecha_world = asignar_manos(result)

            h, w = frame.shape[:2]
            if lienzo is None:
                lienzo = Lienzo(w, h)  # imagen blanca del mismo tamaño que la cámara

            # Punta del índice derecho suavizada (se usa para pintar y para el cursor)
            punto = None
            if mano_derecha is not None:
                frames_sin_derecha = 0
                punta = mano_derecha[indice_id]
                punto = (filtro_x(punta.x, ahora), filtro_y(punta.y, ahora))
            else:
                frames_sin_derecha += 1
                if frames_sin_derecha > FRAMES_TOLERANCIA:
                    # Si la mano se ha ido, al volver el filtro empieza de cero (si no, arrastraría la posición vieja)
                    filtro_x.reset()
                    filtro_y.reset()

            # Si la mano izquierda hace el OK durante varios frames seguidos, el dibujo se da por terminado
            okay_ahora = (mano_izquierda is not None
                          and check_okay(mano_izquierda, mano_izquierda_world))
            frames_okay = frames_okay + 1 if okay_ahora else 0
            dibujo_terminado = frames_okay >= FRAMES_OKAY_NECESARIOS

            # Puño izquierdo cerrado = "modo dibujo" (el pulgar arriba también es un puño,
            # así que se excluye para que no pinte mientras se hace el OK)
            puno_izquierdo = (not okay_ahora
                              and mano_izquierda_world is not None
                              and check_puno_cerrado(mano_izquierda_world))

            # Borrar: puño izquierdo + palma derecha abierta mantenida ~1 segundo
            borrando = (puno_izquierdo
                        and mano_derecha_world is not None
                        and check_palma_abierta(mano_derecha_world))
            frames_borrar = frames_borrar + 1 if borrando else 0
            if frames_borrar >= FRAMES_BORRAR_NECESARIOS:
                lienzo.borrar()  # lienzo en blanco otra vez
                pendientes = []
                lapiz_bajado = False
                frames_borrar = 0

            # ---- Lápiz ----
            # Se dibuja con el índice derecho mientras el puño izquierdo está cerrado
            # (salvo si la derecha está haciendo el gesto de borrar).
            # Para que no salgan rayas raras:
            #  - el lápiz solo baja tras FRAMES_PARA_BAJAR_LAPIZ frames seguidos de puño
            #  - si el puño o la mano "parpadean" unos pocos frames, los puntos se guardan en
            #    "pendientes": si vuelve el puño se pintan (sin huecos), si no se tiran (sin colas)
            #  - un salto brusco del dedo corta el trazo en vez de unir con una línea larga
            #  - los trazos muy cortos (toques accidentales) se descartan al terminar
            quiere_dibujar = puno_izquierdo and not borrando and punto is not None
            if quiere_dibujar:
                frames_puno += 1
                frames_fallo = 0
                if not lapiz_bajado and frames_puno >= FRAMES_PARA_BAJAR_LAPIZ:
                    lapiz_bajado = True
                if lapiz_bajado:
                    for p in pendientes + [punto]:
                        lienzo.anadir_punto(p)
                    pendientes = []
            else:
                frames_puno = 0
                if lapiz_bajado:
                    frames_fallo += 1
                    # Fallo "duro" (gesto de borrar u OK): se levanta el lápiz ya
                    fallo_duro = borrando or okay_ahora
                    if fallo_duro or frames_fallo > FRAMES_TOLERANCIA:
                        if fallo_duro:
                            # Mientras la mano cambiaba de forma (abrir la palma, levantar el pulgar)
                            # aún se pintaba: se quitan esos últimos puntos para que no quede cola
                            del lienzo.trazo_actual[-FRAMES_RECORTE_GESTO:]
                        lapiz_bajado = False
                        pendientes = []
                        lienzo.cerrar_trazo()
                    elif punto is not None:
                        pendientes.append(punto)

            dibujando = lapiz_bajado

            """ #ESTO CONTIENE LA POSICION DE LOS 21 PUNTOS DE LA MANO, EN COORDENADAS NORMALIZADAS (0-1)
            # 0 = BORDE IZQUIERDO; 0 = ARRIBA; 1 = BORDE DERECHO; 1 = ABAJO
            # MULTIPLICAR POR EL ANCHO Y ALTO DE LA IMAGEN PARA OBTENER PIXELES
            if result.hand_landmarks:
                print(result.hand_landmarks[0][indice_id].x, result.hand_landmarks[0][indice_id].y)  # punto 7 del índice"""

            """# ESTO CONTIENE LA POSICION DE LOS 21 PUNTOS DE LA MANO PERO CON DISTANCIA RELATIVA DE LA MANO
            # ES DECIR, ESTEMOS MÁS O MENOS LEJOS DE LA CÁMARA, EL PUNTO 7 DEL ÍNDICE SIEMPRE ESTARÁ EN LA MISMA POSICIÓN RELATIVA A LOS DEMÁS PUNTOS DE LA MANO
            if result.hand_world_landmarks:
                print(result.hand_world_landmarks[0][indice_id].x, result.hand_world_landmarks[0][indice_id].y, result.hand_world_landmarks[0][indice_id].z)  # punto 7 del índice"""

            for hand in result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for p in pts:
                    cv2.circle(frame, p, 4, (0, 0, 255), -1)

            cv2.imshow(VENTANA_CAMARA, frame)

            # Ventana aparte con el lienzo. Se muestra una copia con un cursor
            # en la punta del índice, para que el cursor no se quede pintado
            vista_lienzo = lienzo.imagen.copy()
            if punto is not None:
                cursor = a_pixeles(punto, w, h)
                color_cursor = (255, 0, 0) if dibujando else (150, 150, 150)  # azul = dibujando, gris = no
                cv2.circle(vista_lienzo, cursor, 8, color_cursor, 2)
            if frames_borrar > 0:
                # Barra de progreso roja: si cierras la mano antes de que se llene, se cancela
                progreso = frames_borrar / FRAMES_BORRAR_NECESARIOS
                cv2.putText(vista_lienzo, "Borrando...", (20, h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
                cv2.rectangle(vista_lienzo, (20, h - 35), (w - 20, h - 15), (0, 0, 200), 2)
                cv2.rectangle(vista_lienzo, (20, h - 35), (20 + int((w - 40) * progreso), h - 15), (0, 0, 200), -1)

            if dibujo_terminado:
                # Se guarda la imagen limpia (sin cursor ni textos) y se enseña la confirmación un momento
                lienzo.cerrar_trazo()
                resultado = preparar_salida(lienzo.imagen, tamano, recortar, escala_grises)
                # Trazos originales (normalizados 0-1) -> píxeles del lienzo
                trazos_px = [[(x * w, y * h) for x, y in trazo] for trazo in lienzo.trazos]
                cv2.putText(vista_lienzo, "Dibujo terminado", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 2)
                cv2.imshow(VENTANA_LIENZO, vista_lienzo)
                cv2.waitKey(700)
                break

            cv2.imshow(VENTANA_LIENZO, vista_lienzo)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # Salir también si se cierra cualquiera de las dos ventanas con la X
            if (cv2.getWindowProperty(VENTANA_CAMARA, cv2.WND_PROP_VISIBLE) < 1
                    or cv2.getWindowProperty(VENTANA_LIENZO, cv2.WND_PROP_VISIBLE) < 1):
                break
    finally:
        # Se libera todo aunque haya un error, para poder volver a llamar a la función
        cam.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # en Windows hace falta para que las ventanas se cierren de verdad
        landmarker.close()

    if devolver_trazos:
        return resultado, (trazos_px if resultado is not None else [])
    return resultado


if __name__ == "__main__":
    # Prueba rápida: dibujar y enseñar lo que se devolvería a la red
    imagen = capturar_dibujo()
    if imagen is None:
        print("Dibujo cancelado")
    else:
        print("Dibujo recibido:", type(imagen), imagen.shape, imagen.dtype)
        cv2.imshow("Resultado", imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
