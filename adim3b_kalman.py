"""Adım 3b: Kalman filtresiyle takip (K1 KalmanKutu, K2 SORT, K3 ByteTrack)."""
import numpy as np

# Kalman ayarları (piksel, kare başına) - K4'te video ile ayarlanacak
OLCUM_SAPMA_MERKEZ = 3.0    # R: YOLO kutu merkezinin titremesi (cx, cy)
OLCUM_SAPMA_BOYUT = 6.0     # R: YOLO kutu boyutunun titremesi (w, h)
IVME_SAPMA_KONUM = 1.2     # Q: insanın kare başına hız değişimi (cx, cy)
IVME_SAPMA_BOYUT = 2.5      # Q: kutu büyüme hızındaki değişim (w, h)
BASLANGIC_HIZ_SAPMA = 20.0  # P: ilk karede hız bilinmez, ± en hızlı yürüyüş

def kutudan_olcum(kutu):
    """(x1, y1, x2, y2) köşe kutusunu Kalman'ın kullandığı (cx, cy, w, h)'ye çevirir."""
    x1, y1, x2, y2 = kutu[:4]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2- y1], dtype = float)

def olcumden_kutu(z):
    """(cx, cy, w, h) -> (x1, y1, x2, y2); çizim ve IoU için geri çevirir."""
    cx,cy,w,h = z[:4]
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)

class KalmanKutu:
    """Tek bir kişinin kutusunu sabit hız modeliyle izler."""

    def __init__(self , kutu):
        # Durum: [cx, cy, w, h, vx, vy, vw, vh]; hızlar başta bilinmiyor -> 0
        self.x = np.zeros(8)
        self.x[:4] = kutudan_olcum(kutu)

        # F: hareket modeli -> yeni konum = konum + hız, hız aynı kalır
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)

        # H: ölçüm modeli -> YOLO sadece ilk 4'ü (cx, cy, w, h) görür, hızı görmez
        self.H = np.eye(4,8)

        # Belirsizlikler standart sapma olarak (piksel), karesi = varyans
        # P: durumun şu anki belirsizliği; konumu YOLO'dan biliyoruz, hızı hiç bilmiyoruz
         # R: ölçüm gürültüsü (YOLO titremesi), duran kişide ölçülür
        olcum_sapma = np.array([OLCUM_SAPMA_MERKEZ, OLCUM_SAPMA_MERKEZ,
                                OLCUM_SAPMA_BOYUT, OLCUM_SAPMA_BOYUT])
        self.R = np.diag(olcum_sapma ** 2)

        # Q: süreç gürültüsü, ivmeden türetilir: Δkonum = ½·a, Δhız = a
        G = np.vstack([0.5 * np.eye(4), np.eye(4)])
        ivme_sapma = np.array([IVME_SAPMA_KONUM, IVME_SAPMA_KONUM,
                               IVME_SAPMA_BOYUT, IVME_SAPMA_BOYUT])
        self.Q = G @ np.diag(ivme_sapma ** 2) @ G.T

        # P: başlangıç belirsizliği; konum ilk ölçümden geldi (= R), hız bilinmiyor
        self.P = np.zeros((8, 8))
        self.P[:4, :4] = self.R
        self.P[4:, 4:] = np.eye(4) * BASLANGIC_HIZ_SAPMA ** 2


    def tahmin(self):
        """Bir kare ileri: kutuyu hızıyla kaydırır, belirsizliği büyütür."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2:4] = np.maxmimum(self.x[2:4] ,1.0)  # kutu küçülerek eksi boyuta düşmesin
        return olcumden_kutu(self.x)


    