import sys
import cv2
from ultralytics import YOLO
from adim1_kamera import kaynak_ac
from adim2_tespit import kisileri_bul
from adim3_takip import Takipci, cift_kutulari_ayikla

PENCERE = "Valiz 2 - Sahip"
MIN_ALAN_ORANI = 0.05   # karenin %5'inden küçük kutular otomatik seçilmez


def otomatik_sec(gorunenler, kare_gen, kare_yuk):
    """En büyük ve ortaya en yakın kişinin ID'sini döndürür (yoksa None)."""
    kare_alani = kare_gen * kare_yuk
    orta_x = kare_gen / 2

    en_iyi_id = None
    en_iyi_puan = 0
    for iz_id, (x1, y1, x2, y2, guven) in gorunenler.items():
        alan = (x2 - x1) * (y2 - y1)
        if alan < MIN_ALAN_ORANI * kare_alani:
            continue

        cx = (x1 + x2) / 2
        ortaya_yakinlik = 1 - abs(cx - orta_x) / orta_x
        puan = alan * ortaya_yakinlik

        if puan > en_iyi_puan:
            en_iyi_puan = puan
            en_iyi_id = iz_id
    return en_iyi_id


def tiklanani_bul(gorunenler, nokta):
    """Tıklanan noktayı içeren en küçük kutunun ID'si (yoksa None)."""
    x, y = nokta
    secilen = None
    en_kucuk_alan = None
    for iz_id, (x1, y1, x2, y2, guven) in gorunenler.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            alan = (x2 - x1) * (y2 - y1)
            if en_kucuk_alan is None or alan < en_kucuk_alan:
                en_kucuk_alan = alan
                secilen = iz_id
    return secilen


def fare(olay, x, y, bayraklar, durum):
    """OpenCV her fare olayında bunu çağırır. Sol tıkı kaydediyoruz."""
    if olay == cv2.EVENT_LBUTTONDOWN:
        durum["tik"] = (x, y)


def ciz(kare, gorunenler, sahip_id):
    for iz_id, (x1, y1, x2, y2, guven) in gorunenler.items():
        if iz_id == sahip_id:
            cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(kare, f"SAHIP (ID {iz_id})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.rectangle(kare, (x1, y1), (x2, y2), (160, 160, 160), 1)
            cv2.putText(kare, f"ID {iz_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    if sahip_id is None:
        yazi, renk = "Sahip yok - birine tikla", (0, 255, 255)
    elif sahip_id not in gorunenler:
        yazi, renk = f"SAHIP KAYIP (ID {sahip_id})", (0, 0, 255)
    else:
        yazi, renk = f"Sahip: ID {sahip_id}", (0, 255, 0)
    cv2.putText(kare, yazi, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, renk, 2)


def main():
    kaynak = sys.argv[1] if len(sys.argv) > 1 else "0"
    cap = kaynak_ac(kaynak)
    model = YOLO("yolo11n.pt")
    takipci = Takipci()

    cv2.namedWindow(PENCERE)
    durum = {"tik": None}
    cv2.setMouseCallback(PENCERE, fare, durum)

    sahip_id = None

    while True:
        ok, kare = cap.read()
        if not ok:
            break

        kisiler = kisileri_bul(model, kare)
        kisiler = cift_kutulari_ayikla(kisiler)
        gorunenler = takipci.guncelle(kisiler)

        # Tıklama varsa: tıklanan kişiyi sahip yap
        if durum["tik"] is not None:
            secilen = tiklanani_bul(gorunenler, durum["tik"])
            if secilen is not None:
                sahip_id = secilen
            durum["tik"] = None

        # Sahip hiç seçilmemişse: otomatik seç
        if sahip_id is None:
            h, w = kare.shape[:2]
            sahip_id = otomatik_sec(gorunenler, w, h)

        ciz(kare, gorunenler, sahip_id)
        cv2.imshow(PENCERE, kare)

        tus = cv2.waitKey(1) & 0xFF
        if tus == ord("q"):
            break
        if tus == ord("r"):
            sahip_id = None     # sahibi bırak, bir sonraki karede yeniden otomatik seçilir

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()