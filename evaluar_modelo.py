import urllib.request
import json
import matplotlib.pyplot as plt
import numpy as np
import beit_classifier as beit

CLASES = ["apple", "car", "cat", "tree", "cup"]
MUESTRAS_POR_CLASE = 6

def descargar_dibujo_test(clase):
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{clase}.ndjson"
    req = urllib.request.Request(url, headers={"Range": "bytes=25000000-25020000"})
    dibujos = []

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            contenido = resp.read().decode("utf-8", errors="ignore")
            lineas = contenido.strip().split("\n")[1:-1]
            for linea in lineas:
                if not linea:
                    continue
                d = json.loads(linea)
                trazos = [list(zip(s[0], s[1])) for s in d["drawing"]]
                dibujos.append(trazos)
                if len(dibujos) >= MUESTRAS_POR_CLASE:
                    break
    except Exception as e:
        print(f"Error descargando {clase}: {e}")

    return dibujos

def main():
    print("Cargando clasificador...")
    clf = beit.get_classifier()

    aciertos = 0
    total = 0
    confianzas = []

    print("Evaluando dibujos del conjunto de test...")
    for clase in CLASES:
        dibujos = descargar_dibujo_test(clase)
        for trazos in dibujos:
            pred = clf.predict_strokes(trazos, top_k=1)
            if pred:
                nombre_pred, conf = pred[0]
                confianzas.append(conf * 100)
                total += 1
                if nombre_pred.lower() == clase.lower():
                    aciertos += 1

    precision = (aciertos / total) * 100
    confianza_media = np.mean(confianzas)

    print("\n--- RESULTADOS DEL TEST ---")
    print(f"Total dibujos evaluados: {total}")
    print(f"Aciertos: {aciertos} de {total}")
    print(f"Precision: {precision:.1f}%")
    print(f"Confianza media: {confianza_media:.1f}%")

    metricas = ["Precision (%)", "Confianza (%)"]
    valores = [precision, confianza_media]

    plt.figure(figsize=(6, 4))
    barras = plt.bar(metricas, valores, color=["#3b82f6", "#10b981"], width=0.4)
    plt.ylim(0, 100)
    plt.title("Evaluación con Datos de Test", fontweight="bold")
    plt.ylabel("Porcentaje")

    for bar, val in zip(barras, valores):
        plt.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("grafica_resultados.png")
    print("Grafica actualizada en 'grafica_resultados.png'")

if __name__ == "__main__":
    main()
