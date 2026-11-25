
import sys
import cv2
from ultralytics import YOLO
from adim1_kamera import kaynak_ac
from adim2_tespit import kisileri_bul
from adim3_takip import Takipci, cift_kutulari_ayikla
from adim4_sahip import otomatik_sec, tiklanani_bul, fare, ciz as sahip_ciz

PENCERE = "Valiz 2 - Konum"
PENCERE = "Valiz 2 - Konum"
KISI_BOYU_M = 1.70    # ortalama insan boyu (metre)
KENAR_PAYI = 3        # kutu kenara bu kadar pikselden yakınsa "kesik"
YUMUSATMA = 0.3       # yeni ölçümün ağırlığı (0-1). Küçükse daha sakin ama daha geç tepki

def yatay_hata(kutu, kare_gen):
    """-1 (tam sol) ... 0 (orta) ... +1 (tam sağ)"""
    x1, y1, x2, y2 = kutu[:4]
    cx = (x1 + x2) / 2
    orta = kare_gen / 2
    return (cx - orta) / orta

def kesik_mi(kutu, kare_yuk):
    """Kutu üst ya da alt kenara değiyorsa kişi tam görünmüyordur."""
    return kutu[1] <= KENAR_PAYI or kutu[3] >= kare_yuk - KENAR_PAYI

def mesafe_tahmin(kutu, odak_px):
    """İğne deliği modeli: Z = f * gerçek_boy / piksel_boy (metre)"""
    piksel_boy = kutu[3] - kutu[1]
    if piksel_boy <= 0:
        return None
    return odak_px * KISI_BOYU_M / piksel_boy

def yumusat(eski, yeni):
    if eski is None:
        return yeni
    return (1 - YUMUSATMA)*eski + YUMUSATMA*yeni

def konum_ciz(kare, kutu, hata, mesafe, kesik):
    h, w = kare.shape[:2]
    cx = int((kutu[0] + kutu[2]) / 2)
    cy = int((kutu[1] + kutu[3]) / 2)

    cv2.line(kare, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)    # ekranın ortası
    cv2.line(kare, (w // 2, cy), (cx, cy), (0, 255, 255), 2)         # ortadan sahibe
    cv2.circle(kare, (cx, cy), 6, (0, 255, 255), -1)                 # sahibin merkezi

    if hata < -0.1:
        yon = "SOL"
    elif hata > 0.1:
        yon = "SAG"
    else:
        yon = "ORTA"
    cv2.putText(kare, f"Yatay: {hata:+.2f} ({yon})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if mesafe is not None:
        if kesik:
            yazi, renk = f"Mesafe: ~{mesafe:.1f} m (kesik, guvenilmez)", (0, 165, 255)
        else:
            yazi, renk = f"Mesafe: ~{mesafe:.1f} m", (0, 255, 255)
        cv2.putText(kare, yazi, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)

def main():
    kaynak = sys.argv[1] if len(sys.argv) > 1 else "0"
    cap = kaynak_ac(kaynak)
    model = YOLO("yolo11n.pt")
    takipci = Takipci()

    cv2.namedWindow(PENCERE)
    durum = {"tik": None}
    cv2.setMouseCallback(PENCERE, fare, durum)

    sahip_id = None
    onceki_sahip = None
    hata_y = None       # yumuşatılmış yatay hata
    mesafe_y = None     # yumuşatılmış mesafe

    while True:
        ok, kare = cap.read()
        if not ok:
            break
        h, w = kare.shape[:2]

        kisiler = kisileri_bul(model, kare)
        kisiler = cift_kutulari_ayikla(kisiler)
        gorunenler = takipci.guncelle(kisiler)

        if durum["tik"] is not None:
            secilen = tiklanani_bul(gorunenler, durum["tik"])
            if secilen is not None:
                sahip_id = secilen
            durum["tik"] = None

        if sahip_id is None:
            sahip_id = otomatik_sec(gorunenler, w, h)

        # Sahip değiştiyse eski yumuşatılmış değerleri unut
        if sahip_id != onceki_sahip:
            hata_y = None
            mesafe_y = None
            onceki_sahip = sahip_id

        sahip_ciz(kare, gorunenler, sahip_id)

        if sahip_id is not None and sahip_id in gorunenler:
            kutu = gorunenler[sahip_id]
            odak_px = float(w)      # IRS Görev 3'teki gibi: f ≈ resim genişliği

            hata_y = yumusat(hata_y, yatay_hata(kutu, w))
            m = mesafe_tahmin(kutu, odak_px)
            if m is not None:
                mesafe_y = yumusat(mesafe_y, m)

            konum_ciz(kare, kutu, hata_y, mesafe_y, kesik_mi(kutu, h))

        cv2.imshow(PENCERE, kare)
        tus = cv2.waitKey(1) & 0xFF
        if tus == ord("q"):
            break
        if tus == ord("r"):
            sahip_id = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

