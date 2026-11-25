import sys
import cv2
from ultralytics import YOLO
from adim1_kamera import kaynak_ac

KISI_SINIFI = 0       # YOLO'da "person"
GUVEN_ESIGI = 0.25    # bundan emin olmadıklarını at

def kisileri_bul(model, kare):
    """Karedeki kişilerin kutularını ve güvenlerini liste olarak döndürür."""
    sonuc = model(kare, classes=[KISI_SINIFI], conf=GUVEN_ESIGI, verbose=False)[0]

    kisiler = []
    for kutu in sonuc.boxes:
        x1,y1,x2,y2 = kutu.xyxy[0].tolist()
        guven = float(kutu.conf[0])
        kisiler.append((int(x1), int(y1), int(x2), int(y2), guven))
    return kisiler

def ciz(kare, kisiler):
    for (x1,y1, x2, y2, guven) in kisiler:
        cv2.rectangle(kare,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(kare, f"{guven:.2f}", (x1,y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(kare, f"Kisi sayisi: {len(kisiler)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def main():
    kaynak = sys.argv[1] if len(sys.argv) > 1 else "0"
    cap = kaynak_ac(kaynak)
    model = YOLO("yolo11n.pt")

    while True:
        ok, kare = cap.read()
        if not ok:
            break
        kisiler = kisileri_bul(model, kare)
        ciz(kare, kisiler)

        cv2.imshow("Valiz 2 - Tespit", kare)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



