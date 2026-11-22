import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import time
import numpy as np

# distancia euclidea entre dos puntos de mediapipe
def dist(a, b):
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))

def ratio_dedo(mano, a, b, c, d):
    recorrido = dist(mano[a], mano[b]) + dist(mano[b], mano[c]) + dist(mano[c], mano[d])
    return dist(mano[a], mano[d]) / recorrido

# comprueba si la mano esta cerrada
def check_puno_cerrado(mano, umbral=0.7):
    dedos = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    return all(ratio_dedo(mano, *dedo) < umbral for dedo in dedos)

def dedos_doblados(mano_world):
    dedos = [(6, 8), (10, 12), (14, 16), (18, 20)]
    return all(dist(mano_world[0], mano_world[punta]) < dist(mano_world[0], mano_world[pip])
               for pip, punta in dedos)

def pulgar_arriba(mano_img, margen=0.25):
    tamano_palma = abs(mano_img[0].y - mano_img[5].y) + 1e-6
    resto_mas_alto = min(lm.y for i, lm in enumerate(mano_img) if i not in (3, 4))
    return (resto_mas_alto - mano_img[4].y) / tamano_palma > margen

# gesto de pulgar arriba para terminar dibujo
def check_okay(mano_img, mano_world):
    return dedos_doblados(mano_world) and pulgar_arriba(mano_img)

# palma abierta para borrar
def check_palma_abierta(mano_world, umbral=0.85):
    dedos = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    return all(ratio_dedo(mano_world, *dedo) > umbral for dedo in dedos)

# filtro para suavizar el temblor del dedo
class FiltroOneEuro:
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
    manos = list(zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness))
    izq = der = (None, None)
    if len(manos) >= 2:
        manos.sort(key=lambda m: m[0][0].x)
        izq, der = manos[0][:2], manos[-1][:2]
    elif len(manos) == 1:
        img, world, lado = manos[0]
        if INVERTIR[lado[0].category_name] == "Right":
            der = (img, world)
        else:
            izq = (img, world)
    return izq[0], izq[1], der[0], der[1]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20)
]
INVERTIR = {"Left": "Right", "Right": "Left"}
indice_id = 8

RUTA_MODELO = r"C:\hand_models\hand_landmarker.task"
COLOR_TRAZO = (0, 0, 0)
GROSOR_TRAZO = 5
FRAMES_PARA_BAJAR_LAPIZ = 3
FRAMES_TOLERANCIA = 4
SALTO_MAXIMO = 0.12
MIN_PUNTOS_TRAZO = 4
FRAMES_RECORTE_GESTO = 4
FRAMES_OKAY_NECESARIOS = 10
FRAMES_BORRAR_NECESARIOS = 30

VENTANA_CAMARA = "Hand detection"
VENTANA_LIENZO = "Lienzo"

def a_pixeles(p, w, h):
    return (int(p[0] * w), int(p[1] * h))

# guarda los trazos y repinta la imagen blanca
class Lienzo:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.imagen = np.full((h, w, 3), 255, dtype=np.uint8)
        self.trazos = []
        self.trazo_actual = []

    def reconstruir(self):
        self.imagen[:] = 255
        for trazo in self.trazos:
            pts = np.array([a_pixeles(p, self.w, self.h) for p in trazo], dtype=np.int32)
            cv2.polylines(self.imagen, [pts], False, COLOR_TRAZO, GROSOR_TRAZO, cv2.LINE_AA)

    def cerrar_trazo(self):
        if len(self.trazo_actual) >= MIN_PUNTOS_TRAZO:
            self.trazos.append(self.trazo_actual)
        self.trazo_actual = []
        self.reconstruir()

    def anadir_punto(self, p):
        if self.trazo_actual and math.dist(p, self.trazo_actual[-1]) > SALTO_MAXIMO:
            self.cerrar_trazo()
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
    img = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY) if escala_grises else imagen_bgr.copy()

    if recortar:
        gris = img if escala_grises else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gris < 255)
        if len(xs) > 0:
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            lado = int(max(x1 - x0, y1 - y0) * (1 + 2 * margen)) + 1
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            pad = lado
            relleno = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                         value=255 if escala_grises else (255, 255, 255))
            x_ini, y_ini = cx - lado // 2 + pad, cy - lado // 2 + pad
            img = relleno[y_ini:y_ini + lado, x_ini:x_ini + lado]

    if tamano is not None:
        img = cv2.resize(img, tamano, interpolation=cv2.INTER_AREA)
    return img

def capturar_dibujo(tamano=None, recortar=False, escala_grises=True, camara=0, devolver_trazos=False,
                    palabra=None, marcador=None, tiempo_max=30):
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
    pendientes = []
    frames_okay = 0
    frames_borrar = 0
    resultado = None
    trazos_px = []
    tiempo_inicio = time.monotonic()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ahora = time.monotonic()
            timestamp = max(timestamp + 1, int(ahora * 1000))

            result = landmarker.detect_for_video(mp_image, timestamp)
            mano_izquierda, mano_izquierda_world, mano_derecha, mano_derecha_world = asignar_manos(result)

            h, w = frame.shape[:2]
            if lienzo is None:
                lienzo = Lienzo(w, h)

            punto = None
            if mano_derecha is not None:
                frames_sin_derecha = 0
                punta = mano_derecha[indice_id]
                punto = (filtro_x(punta.x, ahora), filtro_y(punta.y, ahora))
            elif mano_izquierda is not None:
                frames_sin_derecha = 0
                punta = mano_izquierda[indice_id]
                punto = (filtro_x(punta.x, ahora), filtro_y(punta.y, ahora))
            else:
                frames_sin_derecha += 1
                if frames_sin_derecha > FRAMES_TOLERANCIA:
                    filtro_x.reset()
                    filtro_y.reset()

            # deteccion de gestos
            okay_ahora = (mano_izquierda is not None and check_okay(mano_izquierda, mano_izquierda_world))
            frames_okay = frames_okay + 1 if okay_ahora else 0
            dibujo_terminado = frames_okay >= FRAMES_OKAY_NECESARIOS

            puno_izquierdo = (not okay_ahora and mano_izquierda_world is not None and check_puno_cerrado(mano_izquierda_world))
            borrando = (puno_izquierdo and mano_derecha_world is not None and check_palma_abierta(mano_derecha_world))

            if borrando:
                frames_borrar += 1
                if frames_borrar >= FRAMES_BORRAR_NECESARIOS:
                    lienzo.borrar()
                    pendientes = []
                    lapiz_bajado = False
                    frames_borrar = 0
            else:
                frames_borrar = 0

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
                    fallo_duro = borrando or okay_ahora
                    if fallo_duro or frames_fallo > FRAMES_TOLERANCIA:
                        if fallo_duro:
                            del lienzo.trazo_actual[-FRAMES_RECORTE_GESTO:]
                        lapiz_bajado = False
                        pendientes = []
                        lienzo.cerrar_trazo()
                    elif punto is not None:
                        pendientes.append(punto)

            dibujando = lapiz_bajado

            for hand in result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for p in pts:
                    cv2.circle(frame, p, 4, (0, 0, 255), -1)

            cv2.imshow(VENTANA_CAMARA, frame)

            # copia para pintar la interfaz
            vista_lienzo = lienzo.imagen.copy()
            if punto is not None:
                cursor = a_pixeles(punto, w, h)
                color_cursor = (255, 0, 0) if dibujando else (150, 150, 150)
                cv2.circle(vista_lienzo, cursor, 8, color_cursor, 2)

            tiempo_restante = None
            tiempo_agotado = False
            if tiempo_max:
                transcurrido = time.monotonic() - tiempo_inicio
                tiempo_restante = max(0, int(tiempo_max - transcurrido))
                if tiempo_restante == 0:
                    tiempo_agotado = True

            # barra superior
            if marcador or palabra or tiempo_max:
                cv2.rectangle(vista_lienzo, (0, 0), (w, 36), (248, 248, 248), -1)
                cv2.line(vista_lienzo, (0, 36), (w, 36), (215, 215, 215), 1)

            if marcador:
                cv2.putText(vista_lienzo, marcador, (15, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

            if tiempo_restante is not None:
                texto_tiempo = f"Tiempo: {tiempo_restante}s"
                (tw_t, _), _ = cv2.getTextSize(texto_tiempo, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                color_tiempo = (0, 0, 190) if tiempo_restante <= 5 else (60, 60, 60)
                grosor_tiempo = 2 if tiempo_restante <= 5 else 1
                cv2.putText(vista_lienzo, texto_tiempo, ((w - tw_t) // 2, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_tiempo, grosor_tiempo, cv2.LINE_AA)

            if palabra:
                texto_palabra = f"Dibuja: {palabra.upper()}"
                (tw, th), _ = cv2.getTextSize(texto_palabra, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(vista_lienzo, texto_palabra, (w - tw - 15, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 30, 0), 2, cv2.LINE_AA)

            if frames_borrar > 0:
                progreso = frames_borrar / FRAMES_BORRAR_NECESARIOS
                cv2.putText(vista_lienzo, "Borrando...", (20, h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
                cv2.rectangle(vista_lienzo, (20, h - 35), (w - 20, h - 15), (0, 0, 200), 2)
                cv2.rectangle(vista_lienzo, (20, h - 35), (20 + int((w - 40) * progreso), h - 15), (0, 0, 200), -1)

            if dibujo_terminado or tiempo_agotado:
                lienzo.cerrar_trazo()
                resultado = preparar_salida(lienzo.imagen, tamano, recortar, escala_grises)
                trazos_px = [[(x * w, y * h) for x, y in trazo] for trazo in lienzo.trazos]
                texto_ok = "Tiempo agotado" if tiempo_agotado else "Dibujo terminado"
                color_ok = (0, 0, 180) if tiempo_agotado else (0, 140, 0)
                (ow, oh), _ = cv2.getTextSize(texto_ok, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                ox = (w - ow) // 2
                oy = h // 2
                cv2.rectangle(vista_lienzo, (ox - 15, oy - oh - 15), (ox + ow + 15, oy + 15), (255, 255, 255), -1)
                cv2.rectangle(vista_lienzo, (ox - 15, oy - oh - 15), (ox + ow + 15, oy + 15), color_ok, 2)
                cv2.putText(vista_lienzo, texto_ok, (ox, oy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color_ok, 2, cv2.LINE_AA)
                cv2.imshow(VENTANA_LIENZO, vista_lienzo)
                cv2.waitKey(800)
                break

            cv2.imshow(VENTANA_LIENZO, vista_lienzo)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if (cv2.getWindowProperty(VENTANA_CAMARA, cv2.WND_PROP_VISIBLE) < 1
                    or cv2.getWindowProperty(VENTANA_LIENZO, cv2.WND_PROP_VISIBLE) < 1):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        landmarker.close()

    if devolver_trazos:
        return resultado, (trazos_px if resultado is not None else [])
    return resultado

if __name__ == "__main__":
    imagen = capturar_dibujo()
    if imagen is not None:
        cv2.imshow("Resultado", imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
