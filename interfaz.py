"""Interfaz gráfica del Pictionary (pygame).

Ejecutar:  python interfaz.py   [--rounds 3] [--time 30] [--camara 0]

Hilos:
  - principal: pygame (eventos y dibujado de pantallas)
  - SesionCamara: lee la cámara, detecta manos y gestiona el lienzo
  - partida: el guion del juego (voz, rondas, puntos); cambia `App.estado`
  - carga del modelo BEiT y escucha del micrófono, en hilos aparte
"""
import argparse
import math
import random
import re
import sys
import threading
import time
import traceback
import unicodedata

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# beit_classifier (que importa torch, ~5 s) NO se importa aquí: se importa dentro
# de App._cargar_modelo, en un hilo de fondo, para que la ventana salga enseguida.
from hand_detection import (
    FRAMES_BORRAR_NECESARIOS, FRAMES_OKAY_NECESARIOS, FRAMES_PARA_BAJAR_LAPIZ,
    FRAMES_RECORTE_GESTO, FRAMES_TOLERANCIA, HAND_CONNECTIONS, RUTA_MODELO,
    FiltroOneEuro, Lienzo, asignar_manos, check_okay, check_palma_abierta,
    check_puno_cerrado, indice_id, preparar_salida,
)
from reconocimiento_voz_IA import TextToSpeech
from reconocimiento_voz_usuario import escuchar_usuario
from traducciones import traducir

# ------------------------------------------------------------------ estilo
W, H = 1280, 720
FONDO_1 = (20, 22, 42)
FONDO_2 = (46, 30, 74)
PANEL = (34, 37, 64)
PANEL_2 = (46, 50, 84)
BORDE = (64, 68, 108)
TEXTO = (240, 240, 248)
TENUE = (150, 156, 190)
ACENTO = (255, 196, 61)
JUGADOR = (72, 214, 160)
IA = (255, 104, 128)
PAPEL = (252, 250, 244)
TINTA = (28, 28, 40)

LIENZO_RECT = pygame.Rect(32, 88, 752, 564)
BARRA_GESTOS = pygame.Rect(32, 664, 752, 40)
LATERAL_X, LATERAL_W = 816, 432

GESTOS = [
    ("✊☝", "Dibujar", "Puño izquierdo + índice derecho"),
    ("✊✋", "Borrar", "Puño izquierdo + palma derecha (1 s)"),
    ("👍", "Terminar", "Pulgar izquierdo arriba (o Espacio)"),
]

_fuentes = {}


def fuente(tam, tipo="ui"):
    clave = (tam, tipo)
    if clave not in _fuentes:
        if tipo == "mano":
            f = pygame.font.SysFont("segoeprint,comicsansms", tam, bold=True)
        elif tipo == "negrita":
            f = pygame.font.SysFont("segoeuisemibold,segoeui,arial", tam, bold=True)
        elif tipo == "negra":
            f = pygame.font.SysFont("segoeuiblack,segoeui,arial", tam, bold=True)
        elif tipo == "emoji":
            f = pygame.font.SysFont("segoeuiemoji", tam)
        else:
            f = pygame.font.SysFont("segoeui,arial", tam)
        _fuentes[clave] = f
    return _fuentes[clave]


def texto(surf, txt, f, color, pos, ancla="center"):
    img = f.render(txt, True, color)
    r = img.get_rect(**{ancla: pos})
    surf.blit(img, r)
    return r


def fuente_ajustada(txt, tipo, tam_max, ancho_max, tam_min=18):
    tam = tam_max
    while tam > tam_min and fuente(tam, tipo).size(txt)[0] > ancho_max:
        tam -= 2
    return fuente(tam, tipo)


def mezclar(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


_sombras = {}


def tarjeta(surf, rect, color=PANEL, radio=18, sombra=True, borde=BORDE):
    rect = pygame.Rect(rect)
    if sombra:
        clave = (rect.w, rect.h, radio)
        if clave not in _sombras:
            s = pygame.Surface((rect.w + 24, rect.h + 24), pygame.SRCALPHA)
            for i, alfa in enumerate((18, 26, 36)):
                pygame.draw.rect(s, (0, 0, 0, alfa), (12 - 4 + i * 2, 12 + i * 2, rect.w + 8 - i * 4, rect.h),
                                 border_radius=radio + 4 - i)
            _sombras[clave] = s
        surf.blit(_sombras[clave], (rect.x - 12, rect.y - 6))
    pygame.draw.rect(surf, color, rect, border_radius=radio)
    if borde:
        pygame.draw.rect(surf, borde, rect, 2, border_radius=radio)
    return rect


def anillo(surf, centro, radio, fraccion, color, grosor=10, fondo=PANEL_2):
    cx, cy = centro
    pasos = 120
    for i in range(pasos):
        ang = -math.pi / 2 + 2 * math.pi * i / pasos
        x, y = cx + radio * math.cos(ang), cy + radio * math.sin(ang)
        c = color if i / pasos < fraccion else fondo
        pygame.draw.circle(surf, c, (int(x), int(y)), grosor // 2)


def spinner(surf, centro, radio, t, color=ACENTO, grosor=8):
    for i in range(12):
        ang = t * 5 + i * 0.32
        alfa = i / 12
        x = centro[0] + radio * math.cos(ang)
        y = centro[1] + radio * math.sin(ang)
        pygame.draw.circle(surf, mezclar(PANEL_2, color, alfa), (int(x), int(y)), int(grosor / 2 * (0.5 + alfa)))


def dibujar_trazos(surf, rect, trazos, grosor, color=TINTA):
    """Pinta trazos con coordenadas normalizadas (0..1) dentro de `rect`."""
    for trazo in trazos:
        pts = [(rect.x + x * rect.w, rect.y + y * rect.h) for x, y in trazo]
        if len(pts) >= 2:
            pygame.draw.lines(surf, color, False, pts, grosor)
        for p in pts:
            pygame.draw.circle(surf, color, (int(p[0]), int(p[1])), grosor // 2)


def crear_fondo():
    fondo = pygame.Surface((W, H))
    for y in range(H):
        pygame.draw.line(fondo, mezclar(FONDO_1, FONDO_2, y / H), (0, y), (W, y))
    puntos = pygame.Surface((W, H), pygame.SRCALPHA)
    for x in range(20, W, 40):
        for y in range(20, H, 40):
            pygame.draw.circle(puntos, (255, 255, 255, 14), (x, y), 2)
    fondo.blit(puntos, (0, 0))
    return fondo


class Boton:
    def __init__(self, etiqueta, color=ACENTO, color_texto=FONDO_1, tecla=None, tam=26):
        self.etiqueta, self.color, self.color_texto = etiqueta, color, color_texto
        self.tecla, self.tam = tecla, tam
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.activo = True

    def dibujar(self, surf, rect, raton):
        self.rect = pygame.Rect(rect)
        encima = self.activo and self.rect.collidepoint(raton)
        color = self.color if self.activo else PANEL_2
        if encima:
            color = mezclar(color, (255, 255, 255), 0.18)
        r = self.rect.move(0, -2) if encima else self.rect
        tarjeta(surf, r, color, radio=r.h // 2, borde=None)
        col_txt = self.color_texto if self.activo else TENUE
        if self.tecla:
            t1 = fuente(self.tam, "negrita").render(self.etiqueta, True, col_txt)
            t2 = fuente(15, "negrita").render(self.tecla, True, col_txt)
            ancho = t1.get_width() + 12 + t2.get_width() + 16
            x = r.centerx - ancho // 2
            surf.blit(t1, t1.get_rect(midleft=(x, r.centery)))
            chip = pygame.Rect(0, 0, t2.get_width() + 16, t2.get_height() + 4)
            chip.midleft = (x + t1.get_width() + 12, r.centery)
            pygame.draw.rect(surf, col_txt, chip, 2, border_radius=6)
            surf.blit(t2, t2.get_rect(center=chip.center))
        else:
            texto(surf, self.etiqueta, fuente(self.tam, "negrita"), col_txt, r.center)

    def pulsado(self, pos):
        return self.activo and self.rect.collidepoint(pos)


# ------------------------------------------------------------------ cámara + gestos
class SesionCamara(threading.Thread):
    """Lee la cámara continuamente. Cuando está activa, aplica la misma lógica de
    gestos que hand_detection.capturar_dibujo y va rellenando un Lienzo."""

    def __init__(self, camara=0):
        super().__init__(daemon=True)
        self.camara = camara
        self.lock = threading.Lock()
        self.vivo = True
        self.lista = False
        self.error = None
        self.preview = None
        self.activo = False
        self.terminado = threading.Event()
        self.resultado = None
        self.trazos_px = []
        self.motivo = None
        self.vista = self._vista_vacia()
        self._pedir_fin = False
        self._cancelar = False

    @staticmethod
    def _vista_vacia():
        return {"trazos": [], "actual": [], "cursor": None, "lapiz": False, "mano": False,
                "borrar": 0.0, "okay": 0.0, "clave": (0, 0)}

    def empezar(self, tiempo_max):
        with self.lock:
            self.lienzo = None
            self.filtro_x, self.filtro_y = FiltroOneEuro(), FiltroOneEuro()
            self.frames_sin_mano = 0
            self.lapiz_bajado = False
            self.frames_puno = self.frames_fallo = 0
            self.frames_okay = self.frames_borrar = 0
            self.pendientes = []
            self.borrados = 0
            self.resultado, self.trazos_px, self.motivo = None, [], None
            self._pedir_fin = self._cancelar = False
            self.tiempo_max = tiempo_max
            self.t_inicio = time.monotonic()
            self.vista = self._vista_vacia()
            self.terminado.clear()
            self.activo = True

    def tiempo_restante(self):
        if not self.activo:
            return 0.0
        return max(0.0, self.tiempo_max - (time.monotonic() - self.t_inicio))

    def pedir_fin(self):
        self._pedir_fin = True

    def cancelar(self):
        self._cancelar = True

    def obtener_vista(self):
        with self.lock:
            return self.vista

    def run(self):
        try:
            opciones = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=RUTA_MODELO),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
            landmarker = vision.HandLandmarker.create_from_options(opciones)
        except Exception as e:
            self.error = f"No se pudo cargar el modelo de manos: {e}"
            return
        cam = cv2.VideoCapture(self.camara)
        if not cam.isOpened():
            self.error = f"No se pudo abrir la cámara {self.camara}"
            landmarker.close()
            return
        self.lista = True
        timestamp = 0
        try:
            while self.vivo:
                ok, frame = cam.read()
                if not ok:
                    self.error = "La cámara ha dejado de enviar imagen"
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ahora = time.monotonic()
                timestamp = max(timestamp + 1, int(ahora * 1000))
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)

                if self.activo:
                    with self.lock:
                        self._procesar(result, frame.shape[1], frame.shape[0], ahora)

                h, w = rgb.shape[:2]
                for mano in result.hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in mano]
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(rgb, pts[a], pts[b], ACENTO, 2, cv2.LINE_AA)
                    for p in pts:
                        cv2.circle(rgb, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
                self.preview = rgb
        finally:
            cam.release()
            landmarker.close()

    def _procesar(self, result, w, h, ahora):
        if self.lienzo is None:
            self.lienzo = Lienzo(w, h)
        lienzo = self.lienzo
        izq, izq_world, der, der_world = asignar_manos(result)

        punto = None
        mano_dibujo = der if der is not None else izq
        if mano_dibujo is not None:
            self.frames_sin_mano = 0
            punta = mano_dibujo[indice_id]
            punto = (self.filtro_x(punta.x, ahora), self.filtro_y(punta.y, ahora))
        else:
            self.frames_sin_mano += 1
            if self.frames_sin_mano > FRAMES_TOLERANCIA:
                self.filtro_x.reset()
                self.filtro_y.reset()

        okay_ahora = izq is not None and check_okay(izq, izq_world)
        self.frames_okay = self.frames_okay + 1 if okay_ahora else 0
        gesto_fin = self.frames_okay >= FRAMES_OKAY_NECESARIOS

        puno_izq = not okay_ahora and izq_world is not None and check_puno_cerrado(izq_world)
        borrando = puno_izq and der_world is not None and check_palma_abierta(der_world)

        if borrando:
            self.frames_borrar += 1
            if self.frames_borrar >= FRAMES_BORRAR_NECESARIOS:
                lienzo.borrar()
                self.borrados += 1
                self.pendientes = []
                self.lapiz_bajado = False
                self.frames_borrar = 0
        else:
            self.frames_borrar = 0

        if puno_izq and not borrando and punto is not None:
            self.frames_puno += 1
            self.frames_fallo = 0
            if not self.lapiz_bajado and self.frames_puno >= FRAMES_PARA_BAJAR_LAPIZ:
                self.lapiz_bajado = True
            if self.lapiz_bajado:
                for p in self.pendientes + [punto]:
                    lienzo.anadir_punto(p)
                self.pendientes = []
        else:
            self.frames_puno = 0
            if self.lapiz_bajado:
                self.frames_fallo += 1
                fallo_duro = borrando or okay_ahora
                if fallo_duro or self.frames_fallo > FRAMES_TOLERANCIA:
                    if fallo_duro:
                        del lienzo.trazo_actual[-FRAMES_RECORTE_GESTO:]
                    self.lapiz_bajado = False
                    self.pendientes = []
                    lienzo.cerrar_trazo()
                elif punto is not None:
                    self.pendientes.append(punto)

        tiempo_agotado = time.monotonic() - self.t_inicio >= self.tiempo_max
        if gesto_fin or tiempo_agotado or self._pedir_fin or self._cancelar:
            lienzo.cerrar_trazo()
            if not self._cancelar:
                self.resultado = preparar_salida(lienzo.imagen)
                self.trazos_px = [[(x * w, y * h) for x, y in t] for t in lienzo.trazos]
            self.motivo = ("cancelado" if self._cancelar else "gesto" if gesto_fin
                           else "tiempo" if tiempo_agotado else "tecla")
            self.lapiz_bajado = False
            self.activo = False
            self.terminado.set()

        self.vista = {
            "trazos": [list(t) for t in lienzo.trazos],
            "actual": list(lienzo.trazo_actual),
            "cursor": punto,
            "lapiz": self.lapiz_bajado,
            "mano": result.hand_landmarks != [],
            "borrar": self.frames_borrar / FRAMES_BORRAR_NECESARIOS,
            "okay": min(1.0, self.frames_okay / FRAMES_OKAY_NECESARIOS),
            "clave": (self.borrados, len(lienzo.trazos)),
        }


# ------------------------------------------------------------------ respuestas por voz
AFIRMATIVAS = {"si", "correcto", "exacto", "acertaste", "acertado", "bien", "claro", "efectivamente",
               "vale", "perfecto", "yes", "eso"}
NEGATIVAS = {"no", "incorrecto", "fallaste", "fallado", "mal", "nope", "tampoco", "error"}


def normalizar(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def interpretar_respuesta(texto_voz, pred_es, pred_en):
    """True = la IA acertó, False = falló, None = no se entiende."""
    t = normalizar(texto_voz)
    palabras = set(re.findall(r"[a-z]+", t))
    if palabras & NEGATIVAS:
        return False
    if pred_es and normalizar(pred_es) in t or pred_en and pred_en.lower() in t:
        return True
    if palabras & AFIRMATIVAS:
        return True
    return None


class Abortado(Exception):
    pass


# ------------------------------------------------------------------ aplicación
class App:
    def __init__(self, rondas=3, tiempo=30, camara=0, servicios=True):
        pygame.init()
        pygame.display.set_caption("PICtionary")
        self.pantalla = pygame.display.set_mode((W, H), pygame.SCALED | pygame.RESIZABLE)
        self.reloj = pygame.time.Clock()
        self.fondo = crear_fondo()

        self.rondas, self.tiempo = rondas, tiempo
        self.lock_estado = threading.Lock()
        self.estado, self.t_estado, self.d = "inicio", time.monotonic(), {}
        self.partida_id = 0
        self.primera_partida = True
        self.puntos = [0, 0]
        self.historial = []
        self.ronda = 0
        self.respuesta = None
        self.evento_respuesta = threading.Event()
        self.voz = None
        self.voz_texto = ""
        self.error = None

        self.clf = None
        self.error_modelo = None
        self.cache_trazos = (None, None)

        self.b_jugar = Boton("Jugar", tecla="Enter", tam=30)
        self.b_menos_r, self.b_mas_r = Boton("−", PANEL_2, TEXTO), Boton("+", PANEL_2, TEXTO)
        self.b_menos_t, self.b_mas_t = Boton("−", PANEL_2, TEXTO), Boton("+", PANEL_2, TEXTO)
        self.b_si = Boton("Sí, acertaste", JUGADOR, tecla="S")
        self.b_no = Boton("No", IA, tecla="N")
        self.b_otra = Boton("Jugar otra vez", tecla="Enter")
        self.b_menu = Boton("Menú", PANEL_2, TEXTO, tecla="Esc")

        self.tts = TextToSpeech()
        self.cam = SesionCamara(camara)
        if servicios:
            self.cam.start()
            threading.Thread(target=self._cargar_modelo, daemon=True).start()

    # ---------------- hilos de fondo
    def _cargar_modelo(self):
        try:
            import beit_classifier as beit  # import lento (torch): aquí, en segundo plano
            self.clf = beit.get_classifier()
        except Exception as e:
            traceback.print_exc()
            self.error_modelo = str(e)

    def _cambiar(self, estado, **datos):
        self.estado, self.d, self.t_estado = estado, datos, time.monotonic()

    def ir(self, pid, estado, **datos):
        with self.lock_estado:
            if pid != self.partida_id:
                raise Abortado
            self._cambiar(estado, **datos)

    def comprobar(self, pid):
        if pid != self.partida_id:
            raise Abortado

    def hablar(self, pid, txt):
        self.comprobar(pid)
        self.tts.speak_sync(txt)
        self.comprobar(pid)

    def esperar(self, pid, segundos):
        fin = time.monotonic() + segundos
        while time.monotonic() < fin:
            self.comprobar(pid)
            time.sleep(0.03)

    def empezar_partida(self):
        with self.lock_estado:
            self.partida_id += 1
            pid = self.partida_id
            self.error = None
            self._cambiar("cargando")
        threading.Thread(target=self._hilo_partida, args=(pid,), daemon=True).start()

    def volver_menu(self):
        with self.lock_estado:
            self.partida_id += 1
            if self.cam.activo:
                self.cam.cancelar()
            self.evento_respuesta.set()
            self._cambiar("inicio")

    def _hilo_partida(self, pid):
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        try:
            self._partida(pid)
        except Abortado:
            pass
        except Exception as e:
            traceback.print_exc()
            with self.lock_estado:
                if pid == self.partida_id:
                    self.error = str(e)
                    self._cambiar("inicio")

    def _partida(self, pid):
        self.puntos = [0, 0]
        self.historial = []
        self.ronda = 0

        while self.clf is None:
            if self.error_modelo:
                raise RuntimeError(f"No se pudo cargar la IA: {self.error_modelo}")
            self.esperar(pid, 0.1)
        while not self.cam.lista:
            if self.cam.error:
                raise RuntimeError(self.cam.error)
            self.esperar(pid, 0.1)

        if self.primera_partida:
            self.ir(pid, "instrucciones")
            self.hablar(pid, "¡Bienvenido al Pictionary!")
            self.hablar(pid, "Cierra el puño izquierdo y dibuja con el índice derecho. "
                             "Puño izquierdo y palma derecha abierta para borrar. "
                             "Y pulgar izquierdo arriba cuando termines.")
            self.primera_partida = False

        for ronda in range(1, self.rondas + 1):
            self.ronda = ronda
            objetivo_en = random.choice(self.clf.classes)
            objetivo_es = traducir(objetivo_en)

            self.ir(pid, "ronda")
            self.hablar(pid, f"Ronda {ronda} de {self.rondas}. ¡Prepárate!")
            self.ir(pid, "palabra", palabra=objetivo_es)
            self.hablar(pid, f"Tu palabra a dibujar es: {objetivo_es}.")
            self.ir(pid, "cuenta", palabra=objetivo_es)
            self.esperar(pid, 3.0)

            self.cam.empezar(self.tiempo)
            self.ir(pid, "dibujo", palabra=objetivo_es)
            while not self.cam.terminado.wait(0.1):
                self.comprobar(pid)
                if self.cam.error:
                    raise RuntimeError(self.cam.error)
            self.comprobar(pid)
            vista = self.cam.obtener_vista()
            motivo = self.cam.motivo

            self.ir(pid, "analizando", palabra=objetivo_es, vista=vista)
            if motivo == "tiempo":
                self.hablar(pid, "¡Se acabó el tiempo!")
            t0 = time.monotonic()
            resultados = self.clf.predict_strokes(self.cam.trazos_px) if self.cam.trazos_px else []
            self.esperar(pid, max(0.0, 1.2 - (time.monotonic() - t0)))

            pred_en, confianza = resultados[0] if resultados else (None, 0.0)
            pred_es = traducir(pred_en) if pred_en else None
            top = [(traducir(c), p) for c, p in resultados]
            datos = dict(palabra=objetivo_es, vista=vista, pred=pred_es, conf=confianza, top=top)

            self.ir(pid, "prediccion", **datos)
            if pred_en is None:
                self.hablar(pid, "No he visto ningún dibujo.")
            else:
                self.hablar(pid, f"Creo que has dibujado: {pred_es}, con un {int(confianza * 100)} "
                                 f"por ciento de confianza.")

            # Igual que main.py: SIEMPRE pregunta y escucha al jugador.
            self.respuesta = None
            self.evento_respuesta.clear()
            self.voz, self.voz_texto = None, ""
            self.ir(pid, "pregunta", **datos)
            self.hablar(pid, "Dime, ¿he acertado?")
            threading.Thread(target=self._escuchar, args=(pid, pred_es, pred_en), daemon=True).start()
            while not self.evento_respuesta.wait(0.1):
                self.comprobar(pid)
            self.comprobar(pid)

            # Igual que main.py: si la IA dijo la palabra correcta es acierto pase lo que pase;
            # si no, cuenta lo que haya respondido el jugador.
            prediccion_correcta = pred_en is not None and pred_en.lower() == objetivo_en.lower()
            acierto = prediccion_correcta or bool(self.respuesta)

            if acierto:
                self.puntos[0] += 1
                self.historial.append("j")
            else:
                self.puntos[1] += 1
                self.historial.append("ia")
            self.ir(pid, "resultado", acierto=acierto, palabra=objetivo_es, pred=pred_es)
            if acierto:
                self.hablar(pid, "¡He acertado! Has dibujado genial, te llevas el punto.")
            else:
                self.hablar(pid, f"¡Vaya, no lo he adivinado! Era {objetivo_es}, así que el punto me lo llevo yo.")
            self.esperar(pid, 1.2)

        self.ir(pid, "final")
        self.tts.fin_juego(*self.puntos)

    def _escuchar(self, pid, pred_es, pred_en):
        while pid == self.partida_id and not self.evento_respuesta.is_set():
            self.voz = "escuchando"
            t0 = time.monotonic()
            dicho = escuchar_usuario()
            if self.evento_respuesta.is_set() or pid != self.partida_id:
                return
            if dicho:
                self.voz_texto = dicho
                r = interpretar_respuesta(dicho, pred_es, pred_en)
                if r is not None:
                    self.voz = "entendido"
                    self.responder(r)
                    return
                self.voz = "no_entendido"
            elif time.monotonic() - t0 < 2.5:
                self.voz = "sin_micro"
                return
            else:
                self.voz = "no_entendido"
            time.sleep(0.8)

    def responder(self, valor):
        if self.estado == "pregunta" and not self.evento_respuesta.is_set():
            self.respuesta = valor
            self.evento_respuesta.set()

    # ---------------- bucle principal
    def ejecutar(self):
        while True:
            raton = pygame.mouse.get_pos()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if self._tecla(ev.key) == "salir":
                        return
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self._clic(ev.pos)
            self.dibujar(raton)
            pygame.display.flip()
            self.reloj.tick(60)

    def _tecla(self, k):
        e = self.estado
        if k == pygame.K_F11:
            pygame.display.toggle_fullscreen()
        elif k == pygame.K_ESCAPE:
            if e == "inicio":
                return "salir"
            self.volver_menu()
        elif e == "inicio" and k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.empezar_partida()
        elif e == "final" and k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.empezar_partida()
        elif e == "dibujo" and k == pygame.K_SPACE:
            self.cam.pedir_fin()
        elif e == "pregunta" and k == pygame.K_s:
            self.responder(True)
        elif e == "pregunta" and k == pygame.K_n:
            self.responder(False)
        return None

    def _clic(self, pos):
        e = self.estado
        if e == "inicio":
            if self.b_jugar.pulsado(pos):
                self.empezar_partida()
            elif self.b_menos_r.pulsado(pos):
                self.rondas = max(1, self.rondas - 1)
            elif self.b_mas_r.pulsado(pos):
                self.rondas = min(10, self.rondas + 1)
            elif self.b_menos_t.pulsado(pos):
                self.tiempo = max(10, self.tiempo - 5)
            elif self.b_mas_t.pulsado(pos):
                self.tiempo = min(120, self.tiempo + 5)
        elif e == "pregunta":
            if self.b_si.pulsado(pos):
                self.responder(True)
            elif self.b_no.pulsado(pos):
                self.responder(False)
        elif e == "final":
            if self.b_otra.pulsado(pos):
                self.empezar_partida()
            elif self.b_menu.pulsado(pos):
                self.volver_menu()

    # ---------------- pantallas
    def dibujar(self, raton):
        s = self.pantalla
        s.blit(self.fondo, (0, 0))
        t = time.monotonic() - self.t_estado
        e = self.estado
        if e == "inicio":
            self.p_inicio(s, raton)
        elif e == "cargando":
            self.p_centro(s, "Preparando la partida...", self._texto_carga(), t, spin=True)
        elif e == "instrucciones":
            self.p_instrucciones(s, t)
        elif e in ("ronda", "palabra", "cuenta"):
            self.p_ronda(s, t)
        elif e in ("dibujo", "analizando"):
            self.p_dibujo(s, t)
        elif e in ("prediccion", "pregunta"):
            self.p_prediccion(s, t, raton)
        elif e == "resultado":
            self.p_resultado(s, t)
        elif e == "final":
            self.p_final(s, t, raton)

    def _texto_carga(self):
        if self.clf is None:
            return "Cargando la IA (la primera vez descarga unos 350 MB)"
        if not self.cam.lista:
            return "Abriendo la cámara"
        return ""

    def barra_superior(self, s):
        x = texto(s, "PIC", fuente(30, "mano"), ACENTO, (32, 40), "midleft").right
        texto(s, "tionary", fuente(30, "mano"), TEXTO, (x, 40), "midleft")
        if self.ronda:
            r = pygame.Rect(0, 0, 190, 40)
            r.center = (W // 2, 42)
            tarjeta(s, r, PANEL, radio=20, sombra=False)
            texto(s, f"RONDA {self.ronda} / {self.rondas}", fuente(18, "negrita"), TEXTO, r.center)
        texto(s, "Esc  menú   ·   F11  pantalla completa", fuente(15), TENUE, (W - 32, 42), "midright")

    def marcador(self, s, rect, grande=False):
        cx = rect.centerx
        tam = 64 if grande else 46
        sep = 90 if grande else 70
        texto(s, "TÚ", fuente(16, "negrita"), JUGADOR, (cx - sep, rect.y + 24))
        texto(s, "IA", fuente(16, "negrita"), IA, (cx + sep, rect.y + 24))
        texto(s, str(self.puntos[0]), fuente(tam, "negra"), TEXTO, (cx - sep, rect.y + 24 + tam * 0.8))
        texto(s, str(self.puntos[1]), fuente(tam, "negra"), TEXTO, (cx + sep, rect.y + 24 + tam * 0.8))
        texto(s, ":", fuente(tam, "negra"), TENUE, (cx, rect.y + 24 + tam * 0.75))
        n = self.rondas
        paso = 22
        x0 = cx - (n - 1) * paso / 2
        y = rect.bottom - 20
        for i in range(n):
            if i < len(self.historial):
                c = JUGADOR if self.historial[i] == "j" else IA
                pygame.draw.circle(s, c, (int(x0 + i * paso), y), 7)
            else:
                pygame.draw.circle(s, BORDE, (int(x0 + i * paso), y), 7, 2)

    def preview_camara(self, s, rect, etiqueta=True):
        tarjeta(s, rect, PANEL, radio=18)
        interior = rect.inflate(-12, -12)
        if self.cam.preview is not None:
            img = self.cam.preview
            ih, iw = img.shape[:2]
            esc = max(interior.w / iw, interior.h / ih)
            sw, sh = int(iw * esc), int(ih * esc)
            surf = pygame.image.frombuffer(img.tobytes(), (iw, ih), "RGB")
            surf = pygame.transform.smoothscale(surf, (sw, sh))
            recorte = pygame.Rect((sw - interior.w) // 2, (sh - interior.h) // 2, interior.w, interior.h)
            capa = pygame.Surface(interior.size, pygame.SRCALPHA)
            pygame.draw.rect(capa, (255, 255, 255), capa.get_rect(), border_radius=14)
            capa.blit(surf, (0, 0), recorte, special_flags=pygame.BLEND_RGBA_MIN)
            s.blit(capa, interior)
        else:
            msg = self.cam.error or "Abriendo la cámara..."
            texto(s, msg, fuente(18), IA if self.cam.error else TENUE, interior.center)
        if etiqueta:
            chip = pygame.Rect(interior.x + 10, interior.y + 10, 92, 26)
            pygame.draw.rect(s, (0, 0, 0), chip, border_radius=13)
            pygame.draw.circle(s, IA, (chip.x + 14, chip.centery), 5)
            texto(s, "CÁMARA", fuente(13, "negrita"), TEXTO, (chip.x + 26, chip.centery), "midleft")

    def p_centro(self, s, titulo, sub, t, spin=False):
        if spin:
            spinner(s, (W // 2, H // 2 - 60), 34, t)
        texto(s, titulo, fuente(40, "negrita"), TEXTO, (W // 2, H // 2 + 20))
        if sub:
            texto(s, sub + "." * (int(t * 2) % 4), fuente(20), TENUE, (W // 2, H // 2 + 66))

    def p_inicio(self, s, raton):
        x = W // 2
        f = fuente(104, "mano")
        a, b = f.size("PIC")[0], f.size("tionary")[0]
        izq = x - (a + b) // 2
        texto(s, "PIC", f, ACENTO, (izq, 130), "midleft")
        texto(s, "tionary", f, TEXTO, (izq + a, 130), "midleft")
        texto(s, "Dibuja en el aire con tus manos. La IA intentará adivinarlo.",
              fuente(22), TENUE, (x, 215))

        # ajustes
        ajustes = tarjeta(s, (120, 270, 500, 210), PANEL)
        filas = [("Rondas", str(self.rondas), self.b_menos_r, self.b_mas_r),
                 ("Segundos por ronda", str(self.tiempo), self.b_menos_t, self.b_mas_t)]
        for i, (nombre, valor, bm, bp) in enumerate(filas):
            y = ajustes.y + 60 + i * 90
            texto(s, nombre, fuente(22, "negrita"), TEXTO, (ajustes.x + 32, y), "midleft")
            bm.dibujar(s, (ajustes.right - 200, y - 24, 48, 48), raton)
            texto(s, valor, fuente(30, "negra"), ACENTO, (ajustes.right - 116, y))
            bp.dibujar(s, (ajustes.right - 80, y - 24, 48, 48), raton)
            if i == 0:
                pygame.draw.line(s, BORDE, (ajustes.x + 24, y + 45), (ajustes.right - 24, y + 45), 1)

        self.b_jugar.dibujar(s, (120, 504, 500, 76), raton)

        # estado de carga
        y = 612
        if self.error:
            texto(s, f"Error: {self.error}"[:70], fuente(17), IA, (370, y))
        elif self.error_modelo:
            texto(s, "No se pudo cargar la IA (mira la terminal)", fuente(17), IA, (370, y))
        elif self.clf is None:
            spinner(s, (226, y), 9, time.monotonic(), grosor=4)
            texto(s, "Cargando la IA...", fuente(17), TENUE, (246, y), "midleft")
        else:
            pygame.draw.circle(s, JUGADOR, (290, y), 6)
            texto(s, "IA lista para jugar", fuente(17), JUGADOR, (304, y), "midleft")

        # cámara
        texto(s, "Comprueba que se ven bien tus dos manos", fuente(18, "negrita"), TENUE, (930, 262))
        self.preview_camara(s, pygame.Rect(690, 280, 480, 360))

        # gestos
        ancho = 330
        x0 = W // 2 - (ancho * 3 + 24) // 2
        for i, (ico, nombre, desc) in enumerate(GESTOS):
            r = pygame.Rect(x0 + i * (ancho + 12), 656, ancho, 48)
            pygame.draw.rect(s, PANEL, r, border_radius=12)
            texto(s, ico, fuente(22, "emoji"), TEXTO, (r.x + 14, r.centery), "midleft")
            texto(s, nombre, fuente(16, "negrita"), ACENTO, (r.x + 76, r.centery - 9), "midleft")
            texto(s, desc, fuente(13), TENUE, (r.x + 76, r.centery + 10), "midleft")

    def p_instrucciones(self, s, t):
        self.barra_superior(s)
        texto(s, "Cómo se juega", fuente(52, "mano"), TEXTO, (W // 2, 150))
        texto(s, "Dibuja con las manos delante de la cámara", fuente(22), TENUE, (W // 2, 215))
        ancho, alto = 340, 300
        x0 = W // 2 - (ancho * 3 + 40 * 2) // 2
        for i, (ico, nombre, desc) in enumerate(GESTOS):
            k = ease_out((t - i * 0.25) / 0.6)
            r = pygame.Rect(x0 + i * (ancho + 40), 280 + int((1 - k) * 60), ancho, alto)
            tarjeta(s, r, PANEL)
            texto(s, ico, fuente(72, "emoji"), TEXTO, (r.centerx, r.y + 100))
            texto(s, nombre, fuente(34, "mano"), ACENTO, (r.centerx, r.y + 190))
            texto(s, desc, fuente(17), TENUE, (r.centerx, r.y + 240))

    def p_ronda(self, s, t):
        self.barra_superior(s)
        e = self.estado
        if e == "ronda":
            k = ease_out(t / 0.5)
            f = fuente(int(40 + 60 * k), "mano")
            texto(s, f"Ronda {self.ronda}", f, TEXTO, (W // 2, H // 2 - 20))
            texto(s, f"de {self.rondas}", fuente(26), TENUE, (W // 2, H // 2 + 60))
            self.marcador(s, pygame.Rect(W // 2 - 150, H - 190, 300, 150))
            return
        texto(s, "Tu palabra es", fuente(26), TENUE, (W // 2, 200))
        k = ease_out(t / 0.45) if e == "palabra" else 1.0
        carta = pygame.Rect(0, 0, int(640 * (0.6 + 0.4 * k)), int(200 * (0.6 + 0.4 * k)))
        carta.center = (W // 2, 330)
        tarjeta(s, carta, PAPEL, radio=26, borde=None)
        palabra = self.d.get("palabra", "").upper()
        f = fuente_ajustada(palabra, "mano", int(84 * (0.6 + 0.4 * k)), carta.w - 60)
        texto(s, palabra, f, TINTA, carta.center)
        if e == "cuenta":
            n = 3 - int(t)
            if n > 0:
                frac = t - int(t)
                f = fuente(int(120 - 40 * frac), "negra")
                texto(s, str(n), f, mezclar(ACENTO, FONDO_2, frac * 0.8), (W // 2, 560))
            texto(s, "¡Prepara las manos!", fuente(22), TENUE, (W // 2, 650))

    def p_dibujo(self, s, t):
        self.barra_superior(s)
        analizando = self.estado == "analizando"
        v = self.d["vista"] if analizando else self.cam.obtener_vista()

        # lienzo
        tarjeta(s, LIENZO_RECT, PAPEL, radio=20, borde=None)
        area = LIENZO_RECT.inflate(-8, -8)
        if self.cache_trazos[0] != v["clave"]:
            capa = pygame.Surface(area.size, pygame.SRCALPHA)
            dibujar_trazos(capa, capa.get_rect(), v["trazos"], 7)
            self.cache_trazos = (v["clave"], capa)
        s.blit(self.cache_trazos[1], area)
        dibujar_trazos(s, area, [v["actual"]], 7)

        if not analizando:
            if v["cursor"] is not None:
                cx = int(area.x + v["cursor"][0] * area.w)
                cy = int(area.y + v["cursor"][1] * area.h)
                if area.collidepoint(cx, cy):
                    if v["lapiz"]:
                        pygame.draw.circle(s, ACENTO, (cx, cy), 9)
                        pygame.draw.circle(s, TINTA, (cx, cy), 9, 2)
                    else:
                        pygame.draw.circle(s, (160, 160, 175), (cx, cy), 11, 2)
            if not v["mano"]:
                estado, color = "Pon las manos delante de la cámara", TENUE
            elif v["lapiz"]:
                estado, color = "Dibujando", JUGADOR
            else:
                estado, color = "Lápiz levantado", (120, 124, 150)
            chip = pygame.Rect(area.x + 12, area.y + 12, fuente(15, "negrita").size(estado)[0] + 40, 30)
            pygame.draw.rect(s, (238, 236, 228), chip, border_radius=15)
            pygame.draw.circle(s, color, (chip.x + 15, chip.centery), 6)
            texto(s, estado, fuente(15, "negrita"), (70, 70, 90), (chip.x + 28, chip.centery), "midleft")

            for prog, etiqueta, col in ((v["borrar"], "Borrando...", IA), (v["okay"], "Terminando...", JUGADOR)):
                if prog > 0:
                    barra = pygame.Rect(area.x + 40, area.bottom - 56, area.w - 80, 16)
                    texto(s, etiqueta, fuente(18, "negrita"), col, (barra.x, barra.y - 16), "midleft")
                    pygame.draw.rect(s, (225, 222, 214), barra, border_radius=8)
                    pygame.draw.rect(s, col, (barra.x, barra.y, int(barra.w * prog), barra.h), border_radius=8)
        else:
            velo = pygame.Surface(LIENZO_RECT.size, pygame.SRCALPHA)
            pygame.draw.rect(velo, (20, 22, 42, 150), velo.get_rect(), border_radius=20)
            s.blit(velo, LIENZO_RECT)
            spinner(s, (LIENZO_RECT.centerx, LIENZO_RECT.centery - 30), 32, t)
            texto(s, "La IA está pensando" + "." * (int(t * 3) % 4), fuente(30, "negrita"), TEXTO,
                  (LIENZO_RECT.centerx, LIENZO_RECT.centery + 40))

        # gestos
        ancho = (BARRA_GESTOS.w - 16) // 3
        for i, (ico, nombre, _) in enumerate(GESTOS):
            r = pygame.Rect(BARRA_GESTOS.x + i * (ancho + 8), BARRA_GESTOS.y, ancho, BARRA_GESTOS.h)
            pygame.draw.rect(s, PANEL, r, border_radius=12)
            x_ico = texto(s, ico, fuente(18, "emoji"), TEXTO, (r.x + 14, r.centery), "midleft").right
            txt = nombre + ("  ·  Espacio" if nombre == "Terminar" else "")
            texto(s, txt, fuente(16, "negrita"), TEXTO, (x_ico + 12, r.centery), "midleft")

        # lateral: palabra
        palabra_r = tarjeta(s, (LATERAL_X, 88, LATERAL_W, 140), PANEL)
        texto(s, "DIBUJA", fuente(15, "negrita"), TENUE, (palabra_r.centerx, palabra_r.y + 26))
        palabra = self.d.get("palabra", "").upper()
        texto(s, palabra, fuente_ajustada(palabra, "mano", 52, palabra_r.w - 40), ACENTO,
              (palabra_r.centerx, palabra_r.y + 82))

        # lateral: tiempo + marcador
        caja = tarjeta(s, (LATERAL_X, 244, LATERAL_W, 150), PANEL)
        restante = 0.0 if analizando else self.cam.tiempo_restante()
        centro = (caja.x + 92, caja.centery)
        urgente = restante <= 5 and not analizando
        color = IA if urgente else ACENTO
        anillo(s, centro, 52, restante / max(1, self.tiempo), color)
        tam = 38 + (int(6 * abs(math.sin(t * 6))) if urgente else 0)
        texto(s, str(int(math.ceil(restante))), fuente(tam, "negra"), color, (centro[0], centro[1] - 4))
        texto(s, "seg", fuente(13), TENUE, (centro[0], centro[1] + 26))
        pygame.draw.line(s, BORDE, (caja.x + 184, caja.y + 24), (caja.x + 184, caja.bottom - 24), 1)
        self.marcador(s, pygame.Rect(caja.x + 184, caja.y, caja.w - 184, caja.h))

        # lateral: cámara
        self.preview_camara(s, pygame.Rect(LATERAL_X, 410, LATERAL_W, 294))

    def p_prediccion(self, s, t, raton):
        self.barra_superior(s)
        d = self.d
        # miniatura del dibujo
        marco = tarjeta(s, (80, 110, 480, 360), PAPEL, radio=20, borde=None)
        dibujar_trazos(s, marco.inflate(-8, -8), d["vista"]["trazos"], 5)
        texto(s, "Tenías que dibujar", fuente(18), TENUE, (marco.centerx, marco.bottom + 34))
        texto(s, d["palabra"].upper(), fuente_ajustada(d["palabra"].upper(), "mano", 40, 460),
              ACENTO, (marco.centerx, marco.bottom + 78))

        x = 640
        if d["pred"] is None:
            texto(s, "No he visto ningún dibujo", fuente(40, "negrita"), TEXTO, (x, 200), "midleft")
        else:
            texto(s, "Creo que has dibujado...", fuente(26), TENUE, (x, 140), "midleft")
            k = ease_out((t - 0.3) / 0.5)
            if k > 0:
                pred = d["pred"].upper()
                f = fuente_ajustada(pred, "mano", int(48 + 24 * k), 560)
                texto(s, pred, f, TEXTO, (x, 215), "midleft")
            for i, (nombre, p) in enumerate(d["top"]):
                y = 300 + i * 52
                k = ease_out((t - 0.6 - i * 0.15) / 0.6)
                texto(s, nombre, fuente(19, "negrita" if i == 0 else "ui"),
                      TEXTO if i == 0 else TENUE, (x, y), "midleft")
                barra = pygame.Rect(x, y + 14, 480, 10)
                pygame.draw.rect(s, PANEL_2, barra, border_radius=5)
                pygame.draw.rect(s, ACENTO if i == 0 else BORDE,
                                 (barra.x, barra.y, max(10, int(barra.w * p * k)), barra.h), border_radius=5)
                texto(s, f"{p * k * 100:.0f}%", fuente(17, "negrita"), TEXTO if i == 0 else TENUE,
                      (barra.right + 50, y + 8), "midright")

        if self.estado == "pregunta":
            caja = tarjeta(s, (x - 20, 470, 600, 220), PANEL)
            texto(s, "¿He acertado?", fuente(34, "mano"), TEXTO, (caja.x + 30, caja.y + 42), "midleft")
            estado_voz = {
                None: ("Escuchando...", IA),
                "escuchando": ("Escuchando... di «sí» o «no»", IA),
                "no_entendido": ("No te he entendido, repite o usa los botones", TENUE),
                "sin_micro": ("Micrófono no disponible: usa los botones", TENUE),
                "entendido": (f"Has dicho: «{self.voz_texto}»", JUGADOR),
            }[self.voz]
            y = caja.y + 92
            if self.voz in (None, "escuchando"):
                radio = 7 + 3 * abs(math.sin(t * 4))
                pygame.draw.circle(s, IA, (caja.x + 40, y), int(radio))
            texto(s, "🎤", fuente(20, "emoji"), TEXTO, (caja.x + 56, y), "midleft")
            texto(s, estado_voz[0], fuente(17), estado_voz[1], (caja.x + 90, y), "midleft")
            if self.voz_texto and self.voz == "no_entendido":
                texto(s, f"(he oído: «{self.voz_texto}»)", fuente(15), TENUE, (caja.x + 90, y + 24), "midleft")
            self.b_si.dibujar(s, (caja.x + 30, caja.bottom - 78, 300, 56), raton)
            self.b_no.dibujar(s, (caja.x + 350, caja.bottom - 78, 220, 56), raton)

    def p_resultado(self, s, t):
        self.barra_superior(s)
        d = self.d
        k = ease_out(t / 0.5)
        color = JUGADOR if d["acierto"] else IA
        r = pygame.Rect(0, 0, int(760 * (0.85 + 0.15 * k)), int(330 * (0.85 + 0.15 * k)))
        r.center = (W // 2, 300)
        tarjeta(s, r, PANEL, radio=28, borde=color)
        titulo = "¡La IA ha acertado!" if d["acierto"] else "¡La IA no lo ha adivinado!"
        texto(s, titulo, fuente(int(46 * (0.85 + 0.15 * k)), "mano"), color, (r.centerx, r.y + 80))
        texto(s, "Era", fuente(20), TENUE, (r.centerx, r.y + 150))
        texto(s, d["palabra"].upper(), fuente_ajustada(d["palabra"].upper(), "mano", 54, r.w - 60),
              TEXTO, (r.centerx, r.y + 205))
        punto = "+1 punto para ti" if d["acierto"] else "+1 punto para la IA"
        texto(s, punto, fuente(24, "negrita"), color, (r.centerx, r.bottom - 40))
        self.marcador(s, pygame.Rect(W // 2 - 160, 510, 320, 160), grande=True)

    def p_final(self, s, t, raton):
        j, ia = self.puntos
        if j > ia:
            titulo, sub, color = "¡Me has ganado!", "Bien jugado, dibujas genial", JUGADOR
        elif ia > j:
            titulo, sub, color = "¡Te he ganado!", "La próxima vez dibuja más claro ;)", IA
        else:
            titulo, sub, color = "¡Empate!", "Qué partida más reñida", ACENTO
        k = ease_out(t / 0.6)
        texto(s, "FIN DE LA PARTIDA", fuente(18, "negrita"), TENUE, (W // 2, 110))
        texto(s, titulo, fuente(int(50 + 40 * k), "mano"), color, (W // 2, 200))
        texto(s, sub, fuente(22), TENUE, (W // 2, 280))
        caja = tarjeta(s, (W // 2 - 220, 320, 440, 190), PANEL)
        self.marcador(s, caja, grande=True)
        self.b_otra.dibujar(s, (W // 2 - 330, 560, 400, 70), raton)
        self.b_menu.dibujar(s, (W // 2 + 90, 560, 240, 70), raton)


def main():
    parser = argparse.ArgumentParser(description="Pictionary con IA (interfaz gráfica)")
    parser.add_argument("--rounds", "-r", type=int, default=3, help="Número de rondas")
    parser.add_argument("--time", "-t", type=int, default=30, help="Segundos por ronda")
    parser.add_argument("--camara", "-c", type=int, default=0, help="Índice de la cámara")
    args = parser.parse_args()
    app = App(args.rounds, args.time, args.camara)
    try:
        app.ejecutar()
    finally:
        app.partida_id += 1
        app.cam.vivo = False
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
