import sys
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from adim1_kamera import kaynak_ac
from adim2_tespit import kisileri_bul
from adim3_takip import Takipci, cift_kutulari_ayikla, kapsama
from adim4_sahip import otomatik_sec, tiklanani_bul, fare, ciz as sahip_ciz
from adim5_konum import yatay_hata, kesik_mi, mesafe_tahmin, yumusat, konum_ciz
from adim6_hareket import hareket_karari, hareket_ciz


PENCERE = "Valiz 2 - OSNet"
MODEL_YOLU = "modeller/osnet_x0_25_msmt17.onnx"
BENZERLIK_ESIGI = 0.7   # kosinüs benzerliği; histogramdakinden farklı bir ölçek, test edip ayarlayacağız
ONAY_KARE = 5
GALERI_BOYUTU = 10
GALERI_ARALIK = 10

# OSNet eğitilirken resimler bu ortalama ve sapmayla normalize edilmiş (ImageNet değerleri, RGB sırası)
ORTALAMA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SAPMA = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def baska_kisiyle_ortusuyor(sahip_kutu, gorunenler, sahip_id):
    """Sahibin kutusu başka biriyle örtüşüyorsa galeriye örnek eklemeyelim."""
    for iz_id, kutu in gorunenler.items():
        if iz_id != sahip_id and kapsama(sahip_kutu, kutu) > 0.3:
            return True
    return False

def osnet_yukle(yol):
    return ort.InferenceSession(yol, providers=["CPUExecutionProvider"])

def gorunum_cikar(oturum, kare, kutu):
    """Kişinin kutusundan 512 elemanlı, uzunluğu 1 olan bir vektör. Olmazsa None."""
    x1, y1, x2, y2 = kutu[:4]
    parca = kare[y1:y2, x1:x2]
    if parca.shape[0] < 4 or parca.shape[1] < 4:
        return None

    # 1) Ağın beklediği boyuta getir: 128 genişlik x 256 yükseklik
    parca = cv2.resize(parca, (128, 256))
    # 2) BGR -> RGB, 0-255 -> 0-1
    parca = cv2.cvtColor(parca, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # 3) Eğitimdeki gibi normalize et
    parca = (parca - ORTALAMA) / SAPMA
    # 4) (yükseklik, genişlik, kanal) -> (1, kanal, yükseklik, genişlik)
    girdi = parca.transpose(2, 0, 1)[np.newaxis]

    vektor = oturum.run(None, {"x": girdi})[0][0]       # 512 sayı
    return vektor / (np.linalg.norm(vektor) + 1e-6)    # uzunluğu 1 yap


def benzerlik(a, b):
    """Kosinüs benzerliği: ikisi de uzunluk 1 olduğu için sadece nokta çarpımı."""
    return float(np.dot(a, b))


def galeriye_benzerlik(galeri, ozellik):
    if not galeri or ozellik is None:
        return 0.0
    return max(benzerlik(g, ozellik) for g in galeri)


def reid_ciz(kare, gorunenler, benzerlikler, galeri_boyu, aday_id, aday_sayac):
    for iz_id, b in benzerlikler.items():
        x1, y1, x2, y2, guven = gorunenler[iz_id]
        renk = (0, 255, 0) if b >= BENZERLIK_ESIGI else (0, 0, 255)
        cv2.putText(kare, f"b={b:.2f}", (x1 + 5, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)

    yazi = f"OSNet  Galeri: {galeri_boyu}"
    if aday_id is not None:
        yazi += f"   Aday: ID {aday_id} ({aday_sayac}/{ONAY_KARE})"
    cv2.putText(kare, yazi, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    kaynak = sys.argv[1] if len(sys.argv) > 1 else "0"
    cap = kaynak_ac(kaynak)
    model = YOLO("yolo11n.pt")
    oturum = osnet_yukle(MODEL_YOLU)
    takipci = Takipci()

    cv2.namedWindow(PENCERE)
    durum = {"tik": None}
    cv2.setMouseCallback(PENCERE, fare, durum)

    sahip_id = None
    onceki_sahip = None
    hata_y = None
    mesafe_y = None

    galeri = []
    kare_no = 0
    aday_id = None
    aday_sayac = 0

    while True:
        ok, kare = cap.read()
        if not ok:
            break
        h, w = kare.shape[:2]
        kare_no += 1

        kisiler = kisileri_bul(model, kare)
        kisiler = cift_kutulari_ayikla(kisiler)
        gorunenler = takipci.guncelle(kisiler)

        if durum["tik"] is not None:
            secilen = tiklanani_bul(gorunenler, durum["tik"])
            if secilen is not None:
                sahip_id = secilen
                galeri = []
            durum["tik"] = None

        if sahip_id is None:
            sahip_id = otomatik_sec(gorunenler, w, h)
            galeri = []

        benzerlikler = {}
        if galeri:
            for iz_id, kutu in gorunenler.items():
                benzerlikler[iz_id] = galeriye_benzerlik(galeri, gorunum_cikar(oturum, kare, kutu))

        sahip_gorunuyor = sahip_id is not None and sahip_id in gorunenler

        if sahip_gorunuyor:
            aday_id, aday_sayac = None, 0
            kutu = gorunenler[sahip_id]
            if (not galeri or kare_no % GALERI_ARALIK == 0) and \
                    not baska_kisiyle_ortusuyor(kutu, gorunenler, sahip_id):
                oz = gorunum_cikar(oturum, kare, kutu)
                if oz is not None:
                    galeri.append(oz)
                    if len(galeri) > GALERI_BOYUTU:
                        galeri.pop(0)

        elif sahip_id is not None and benzerlikler:
            en_iyi_id = max(benzerlikler, key=benzerlikler.get)
            if benzerlikler[en_iyi_id] >= BENZERLIK_ESIGI:
                if en_iyi_id == aday_id:
                    aday_sayac += 1
                else:
                    aday_id, aday_sayac = en_iyi_id, 1
                if aday_sayac >= ONAY_KARE:
                    print(f"Sahip geri geldi: ID {sahip_id} -> ID {en_iyi_id}")
                    sahip_id = en_iyi_id
                    aday_id, aday_sayac = None, 0
            else:
                aday_id, aday_sayac = None, 0

        if sahip_id != onceki_sahip:
            hata_y = None
            mesafe_y = None
            onceki_sahip = sahip_id

        sahip_ciz(kare, gorunenler, sahip_id)
        if sahip_id is not None and sahip_id in gorunenler:
            kutu = gorunenler[sahip_id]
            hata_y = yumusat(hata_y, yatay_hata(kutu, w))
            m = mesafe_tahmin(kutu, float(w))
            if m is not None:
                mesafe_y = yumusat(mesafe_y, m)
            konum_ciz(kare, kutu, hata_y, mesafe_y, kesik_mi(kutu, h))

        reid_ciz(kare, gorunenler, benzerlikler, len(galeri), aday_id, aday_sayac)
        
        sahip_var = sahip_id is not None and sahip_id in gorunenler
        kesik = kesik_mi(gorunenler[sahip_id], h) if sahip_var else False
        v_cmd, w_cmd, hareket_durumu = hareket_karari(sahip_var, hata_y, mesafe_y, kesik)
        hareket_ciz(kare, v_cmd, w_cmd, hareket_durumu)
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












