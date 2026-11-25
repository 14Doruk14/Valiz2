import sys
import cv2
from ultralytics import YOLO
from adim1_kamera import kaynak_ac
from adim2_tespit import kisileri_bul

IOU_ESIGI = 0.3   # bunun altındaki örtüşme "farklı kişi"
MAX_KAYIP = 15    # bu kadar kare görülmeyen iz silinir (~0.5 sn)
YENI_ID_ESIGI = 0.5   # yeni kişi açmak için gereken en düşük güven

def iou(a, b):
    """a ve b: (x1, y1, x2, y2, ...) kutular. 0 ile 1 arası örtüşme döndürür."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]

    kx1 = max(ax1, bx1)
    ky1 = max(ay1, by1)
    kx2 = min(ax2, bx2)
    ky2 = min(ay2, by2)

    k_gen = max(0, kx2 - kx1)
    k_yuk = max(0, ky2 - ky1)
    kesisim = k_gen * k_yuk

    alan_a = (ax2 - ax1) * (ay2 - ay1)
    alan_b = (bx2 - bx1) * (by2 - by1)
    birlesim = alan_a + alan_b - kesisim

    if birlesim <= 0:
        return 0.0
    return kesisim / birlesim

def kapsama(a,b):
    """Kesişim / küçük kutunun alanı. Küçük kutu büyüğün içindeyse ~1."""
    kx1 = max(a[0], b[0])
    ky1 = max(a[1], b[1])
    kx2 = min(a[2], b[2])
    ky2 = min(a[3], b[3])
    kesisim = max(0, kx2 - kx1) * max(0, ky2 - ky1)

    alan_a = (a[2] - a[0]) * (a[3] - a[1])
    alan_b = (b[2] - b[0]) * (b[3] - b[1])
    kucuk = min(alan_a, alan_b)
    if kucuk <= 0:
        return 0.0
    return kesisim / kucuk

def cift_kutulari_ayikla(kisiler, esik=0.7):
    """Aynı kişiye düşmüş iç içe kutulardan sadece en güvenli olanı bırakır."""
    kisiler = sorted(kisiler, key = lambda k : k[4], reverse=True) # güvene göre, büyükten küçüğe
    kalan = []
    for kisi in kisiler:
        ayni_kisi = False
        for secilen in kalan:
            if kapsama(kisi, secilen) >= esik:
                ayni_kisi = True
                break
        if not ayni_kisi:
            kalan.append(kisi)
    return kalan
    



class Takipci:
    def __init__(self):
        self.izler = {}        # {id: {"kutu": (x1,y1,x2,y2,guven), "kayip": 0}}
        self.sonraki_id = 1

    def guncelle(self, kisiler):
        # 1) Tüm iz-tespit çiftleri için IoU hesapla, eşiği geçenleri topla
        adaylar = []
        for iz_id, iz in self.izler.items():
            for i, kisi in enumerate(kisiler):
                skor = iou(iz["kutu"], kisi)
                if skor >= IOU_ESIGI:
                    adaylar.append((skor, iz_id, i))

        # 2) En yüksek IoU'dan başlayarak eşleştir (açgözlü)
        adaylar.sort(reverse=True)
        eslesen_izler = set()
        eslesen_kisiler = set()
        for skor, iz_id, i in adaylar:
            if iz_id in eslesen_izler or i in eslesen_kisiler:
                continue
            self.izler[iz_id]["kutu"] = kisiler[i]
            self.izler[iz_id]["kayip"] = 0
            eslesen_izler.add(iz_id)
            eslesen_kisiler.add(i)

        # 3) Eşleşmeyen izler: kayıp sayacını artır, çok uzun kayıpsa sil
        for iz_id in list(self.izler):
            if iz_id not in eslesen_izler:
                self.izler[iz_id]["kayip"] += 1
                if self.izler[iz_id]["kayip"] > MAX_KAYIP:
                    del self.izler[iz_id]

        # 4) Eşleşmeyen tespitler: yeni kişi, yeni ID
        for i, kisi in enumerate(kisiler):
            if i not in eslesen_kisiler and kisi[4] >= YENI_ID_ESIGI:
                self.izler[self.sonraki_id] = {"kutu": kisi, "kayip": 0}
                self.sonraki_id += 1

        # Sadece bu karede görülenleri döndür: {id: kutu}
        sonuc = {}
        for iz_id, iz in self.izler.items():
            if iz["kayip"] == 0:
                sonuc[iz_id] = iz["kutu"]
        return sonuc


def id_rengi(iz_id):
    """Her ID'ye sabit, birbirinden farklı bir renk."""
    return ((iz_id * 37) % 256, (iz_id * 91) % 256, (iz_id * 151) % 256)


def ciz_izler(kare, gorunenler):
    for iz_id, (x1, y1, x2, y2, guven) in gorunenler.items():
        renk = id_rengi(iz_id)
        cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 2)
        cv2.putText(kare, f"ID {iz_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)


def main():
    kaynak = sys.argv[1] if len(sys.argv) > 1 else "0"
    cap = kaynak_ac(kaynak)
    model = YOLO("yolo11n.pt")
    takipci = Takipci()

    while True:
        ok, kare = cap.read()
        if not ok:
            break

        kisiler = kisileri_bul(model, kare)
        kisiler = cift_kutulari_ayikla(kisiler)
        gorunenler = takipci.guncelle(kisiler)
        ciz_izler(kare, gorunenler)

        cv2.imshow("Valiz 2 - Takip", kare)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()