import cv2
import numpy as np
import matplotlib.pyplot as plt
import beit_classifier as beit

# Categorias a evaluar
PALABRAS = [
    "apple", "car", "cat", "tree", "cup",
    "house", "fish", "star", "flower", "book",
    "sun", "moon", "hat", "shoe", "chair",
    "key", "clock", "guitar", "umbrella", "bicycle",
    "dog", "bird", "snake", "spider", "butterfly",
    "pizza", "cake", "ice cream", "banana", "mushroom",
    "airplane", "bus", "truck", "boat", "helicopter",
    "sword", "crown", "diamond", "heart", "lightning",
    "eye", "hand", "skull", "tooth", "ear",
    "door", "window", "ladder", "bridge", "fence",
    "pencil", "scissors", "envelope", "candle", "camera",
    "mountain", "cloud", "tornado", "snowflake", "rainbow",
]

ANCHO, ALTO = 500, 500
dibujando = False
trazos = []
trazo_actual = []


def on_mouse(event, x, y, flags, param):
    global dibujando, trazo_actual
    if event == cv2.EVENT_LBUTTONDOWN:
        dibujando = True
        trazo_actual = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and dibujando:
        trazo_actual.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        dibujando = False
        if len(trazo_actual) > 1:
            trazos.append(list(trazo_actual))
        trazo_actual = []


def dibujar_lienzo(palabra, num, total):
    lienzo = np.ones((ALTO, ANCHO, 3), dtype=np.uint8) * 255

 
    for t in trazos:
        pts = np.array(t, dtype=np.int32)
        cv2.polylines(lienzo, [pts], False, (0, 0, 0), 3)

    if len(trazo_actual) > 1:
        pts = np.array(trazo_actual, dtype=np.int32)
        cv2.polylines(lienzo, [pts], False, (0, 0, 0), 3)

   
    texto = f"Dibuja: {palabra} ({num}/{total})"
    cv2.putText(lienzo, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 200), 2)
    cv2.putText(lienzo, "ESPACIO=siguiente  C=borrar  ESC=salir", (10, ALTO - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    return lienzo


def main():
    global trazos, trazo_actual

    print("Cargando clasificador...")
    clf = beit.get_classifier()

    resultados = []

    cv2.namedWindow("Evaluar")
    cv2.setMouseCallback("Evaluar", on_mouse)

    for i, palabra in enumerate(PALABRAS):
        trazos = []
        trazo_actual = []

        while True:
            lienzo = dibujar_lienzo(palabra, i + 1, len(PALABRAS))
            cv2.imshow("Evaluar", lienzo)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                if resultados:
                    generar_grafica(resultados)
                return
            elif key == ord('c') or key == ord('C'):
                trazos = []
                trazo_actual = []
            elif key == 32:  # ESPACIO
                if not trazos:
                    continue
                pred = clf.predict_strokes(trazos, top_k=1)
                if pred:
                    nombre, conf = pred[0]
                    acierto = nombre.lower() == palabra.lower()
                    resultados.append({
                        "palabra": palabra,
                        "prediccion": nombre,
                        "confianza": conf * 100,
                        "acierto": acierto
                    })
                    estado = "OK" if acierto else "FALLO"
                    print(f"  {palabra} -> {nombre} ({conf*100:.1f}%) [{estado}]")
                break

    cv2.destroyAllWindows()
    generar_grafica(resultados)


def generar_grafica(resultados):
    aciertos = sum(1 for r in resultados if r["acierto"])
    total = len(resultados)
    precision = (aciertos / total) * 100
    confianza_media = np.mean([r["confianza"] for r in resultados])

    print(f"\n--- RESULTADOS ---")
    print(f"Dibujos evaluados: {total}")
    print(f"Aciertos: {aciertos}/{total}")
    print(f"Precision: {precision:.1f}%")
    print(f"Confianza media: {confianza_media:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metricas = ["Precision (%)", "Confianza (%)"]
    valores = [precision, confianza_media]
    colores = ["#3b82f6", "#10b981"]
    barras = axes[0].bar(metricas, valores, color=colores, width=0.4)
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Metricas generales", fontweight="bold")
    axes[0].set_ylabel("Porcentaje")
    for bar, val in zip(barras, valores):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 2,
                     f"{val:.1f}%", ha="center", fontweight="bold")

    
    palabras = [r["palabra"] for r in resultados]
    confs = [r["confianza"] for r in resultados]
    colores_p = ["#22c55e" if r["acierto"] else "#ef4444" for r in resultados]
    axes[1].bar(palabras, confs, color=colores_p, width=0.5)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Confianza por dibujo", fontweight="bold")
    axes[1].set_ylabel("Confianza (%)")
    axes[1].tick_params(axis='x', rotation=45)
 
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color="#22c55e", label="Acierto"),
                            Patch(color="#ef4444", label="Fallo")], loc="upper right")

    plt.tight_layout()
    plt.savefig("grafica_resultados.png", dpi=150)
    print("Grafica guardada en 'grafica_resultados.png'")


if __name__ == "__main__":
    main()
