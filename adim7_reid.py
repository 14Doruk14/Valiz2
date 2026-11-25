import sys
import cv2
from ultralytics import YOLO
from adim1_kamera import kaynak_ac
from adim2_tespit import kisileri_bul
from adim3_takip import Takipci, cift_kutulari_ayikla, kapsama
from adim4_sahip import otomatik_sec, tiklanani_bul, fare, ciz as sahip_ciz
from adim5_konum import yatay_hata, kesik_mi, mesafe_tahmin, yumusat, konum_ciz

PENCERE = "Valiz 2 - ReID"
BENZERLIK_ESIGI = 0.6   # bunun üstü "sahibe benziyor"
ONAY_KARE = 5           # üst üste bu kadar kare en benzer kalırsa sahip kabul et
GALERI_BOYUTU = 10      # sahipten en fazla kaç örnek tutulur
GALERI_ARALIK = 10      # kaç karede bir yeni örnek eklenir

def gorunum_cikar(kare, kutu):
    """Kutunun üst ve alt yarısı için H-S renk histogramı. Olmazsa None."""
    x1,y1,x2,y2 = kutu[:4]
    gen = x2 - x1
    yuk = y2 - y1

    # Kenarlardaki arka planı azaltmak için içeri kırp
    x1 = x1 + int(gen * 0.2)
    x2 = x2 + int(gen * 0.2)
    y1 = y1 + int(yuk * 0.1)
    y2 = y1 - int(yuk * 0.1)

    parca = kare[y1:y2 , x1:x2]
    if parca.shape[0] < 4 or parca.shape[1] < 4:
        return None

    hsv = cv2.cvtColor(parca, cv2.COLORBGR2HSV)
    orta = hsv.shape[0] // 2

    ozellik = []
    for yarim in (hsv[:orta], hsv[orta:]):
        # kanal 0 = H (16 kutu, 0-180), kanal 1 = S (8 kutu, 0-256)
        hist = cv2.calHist([yarim], [0,1] , None, [16,8], [0,180,0,256])
        hist = hist / (hist.sum() + 1e-6)
        ozellik.append(hist)
    return ozellik

def benzerlik(a,b):
    """İki görünüm arasında 0 (hiç) ... 1 (aynı) benzerlik."""
    toplam = 0.0
    for ha, hb in zip(a,b):
        uzaklik = cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)   # 0 aynı, 1 tamamen farklı
        toplam += 1 - uzaklik
    return toplam/len(a)


def galeriye_benzerlik(galeri, ozellik):
    """Galerideki örneklerden en benzerine olan benzerlik."""
    if not galeri or ozellik is None:
        return 0.0
    return max(benzerlik(g, ozellik) for g in galeri)

def baska_kisiyle_ortusuyor(sahip_kutu, gorunenler, sahip_id):
    for iz_id, kutu in gorunenler.items():
        if iz_id != sahip_id and kapsama(sahip_kutu, kutu) > 0.3:
            return True
        return False

def reid_ciz(kare, gorunenler, benzerlikler, galeri_boyu, aday_id, aday_sayac):
    for iz_id, b in benzerlikler.items():
        x1, y1, x2, y2, guven = gorunenler[iz_id]
        renk = (0, 255, 0) if b >= BENZERLIK_ESIGI else (0, 0, 255)
        cv2.putText(kare, f"b={b:.2f}", (x1 + 5, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)

    yazi = f"Galeri: {galeri_boyu}"
    if aday_id is not None:
        yazi += f"   Aday: ID {aday_id} ({aday_sayac}/{ONAY_KARE})"
    cv2.putText(kare, yazi, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

        # --- Sahip seçimi (yeni seçimde galeri sıfırlanır) ---
        if durum["tik"] is not None:
            secilen = tiklanani_bul(gorunenler, durum["tik"])
            if secilen is not None:
                sahip_id = secilen
                galeri = []
            durum["tik"] = None

        if sahip_id is None:
            sahip_id = otomatik_sec(gorunenler, w, h)
            galeri = []

        # --- Galeri varsa herkesin sahibe benzerliği ---
        benzerlikler = {}
        if galeri:
            for iz_id, kutu in gorunenler.items():
                benzerlikler[iz_id] = galeriye_benzerlik(galeri, gorunum_cikar(kare, kutu))

        sahip_gorunuyor = sahip_id is not None and sahip_id in gorunenler

        if sahip_gorunuyor:
            # Sahip görünüyor: ara ara galeriye örnek ekle
            aday_id, aday_sayac = None, 0
            kutu = gorunenler[sahip_id]
            if (not galeri or kare_no % GALERI_ARALIK == 0) and \
                    not baska_kisiyle_ortusuyor(kutu, gorunenler, sahip_id):
                oz = gorunum_cikar(kare, kutu)
                if oz is not None:
                    galeri.append(oz)
                    if len(galeri) > GALERI_BOYUTU:
                        galeri.pop(0)       # en eski örneği at

        elif sahip_id is not None and benzerlikler:
            # Sahip kayıp: en benzer kişi üst üste ONAY_KARE kare eşiği geçerse o sahiptir
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

        # --- Konum (Adım 5) ---
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
