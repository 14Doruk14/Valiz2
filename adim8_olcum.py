
import os
import sys
import time
import cv2
from adim1_kamera import kaynak_ac


class Zamanlayici:
    """Her aşamanın kaç ms sürdüğünü ölçer (yumuşatılmış)."""

    def __init__(self, yumusatma=0.1):
        self.yumusatma = yumusatma
        self.sureler = {}          # {"yolo": 35.2, "osnet": 12.0, ...}
        self.onceki = None

    def basla(self):
        """Döngünün en başında çağır: kronometreyi sıfırla."""
        self.onceki = time.perf_counter()

    def tur(self, ad):
        """Son tur()'dan (ya da basla()'dan) bu yana geçen süreyi 'ad' olarak kaydet."""
        simdi = time.perf_counter()
        ms = (simdi - self.onceki) * 1000
        self.onceki = simdi

        eski = self.sureler.get(ad)
        if eski is None:
            self.sureler[ad] = ms
        else:
            self.sureler[ad] = (1 - self.yumusatma) * eski + self.yumusatma * ms

    def toplam(self):
        return sum(self.sureler.values())

    def fps(self):
        t = self.toplam()
        return 1000 / t if t > 0 else 0.0

    def rapor(self):
        print("---- Ortalama süreler ----")
        for ad, ms in self.sureler.items():
            print(f"{ad:8s} {ms:6.1f} ms")
        print(f"{'TOPLAM':8s} {self.toplam():6.1f} ms  ->  {self.fps():.1f} FPS")


def zaman_ciz(kare, zaman):
    """Sağ üst köşeye aşama sürelerini ve FPS'i yazar."""
    x = kare.shape[1] - 190
    y = 25
    for ad, ms in zaman.sureler.items():
        cv2.putText(kare, f"{ad:7s} {ms:5.1f} ms", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += 20
    cv2.putText(kare, f"FPS: {zaman.fps():.1f}", (x, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


class Kaydedici:
    """k tuşuyla aç/kapa video kaydı. Dosyalar kayitlar/video_01.mp4, video_02.mp4 ..."""

    def __init__(self, klasor="kayitlar", fps=30):
        self.klasor = klasor
        self.fps = fps
        self.yazici = None
        self.dosya = None

    def kayitta_mi(self):
        return self.yazici is not None

    def sonraki_dosya(self):
        os.makedirs(self.klasor, exist_ok=True)
        n = 1
        while True:
            yol = os.path.join(self.klasor, f"video_{n:02d}.mp4")
            if not os.path.exists(yol):
                return yol
            n += 1

    def degistir(self, kare):
        """Kayıt kapalıysa başlat, açıksa durdur."""
        if self.yazici is None:
            h, w = kare.shape[:2]
            self.dosya = self.sonraki_dosya()
            kodek = cv2.VideoWriter_fourcc(*"mp4v")
            self.yazici = cv2.VideoWriter(self.dosya, kodek, self.fps, (w, h))
            print("Kayıt başladı:", self.dosya)
        else:
            self.durdur()

    def yaz(self, kare):
        if self.yazici is not None:
            self.yazici.write(kare)

    def durdur(self):
        if self.yazici is not None:
            self.yazici.release()
            print("Kayıt bitti:", self.dosya)
            self.yazici = None


def main():
    kaynak = sys.argv[1] if len(sys.argv) > 1 else "0"
    cap = kaynak_ac(kaynak)

    kamera_fps = cap.get(cv2.CAP_PROP_FPS)
    if kamera_fps <= 0 or kamera_fps > 60:
        kamera_fps = 30
    print("Kamera FPS (kameranın söylediği):", kamera_fps)

    kaydedici = Kaydedici(fps=kamera_fps)
    zaman = Zamanlayici()

    while True:
        zaman.basla()
        ok, kare = cap.read()
        if not ok:
            break
        zaman.tur("kamera")

        kaydedici.yaz(kare)          # ÖNCE ham kareyi yaz, üstüne çizim yapmadan

        if kaydedici.kayitta_mi():
            cv2.circle(kare, (25, 25), 10, (0, 0, 255), -1)
            cv2.putText(kare, "REC", (42, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        zaman_ciz(kare, zaman)

        cv2.imshow("Valiz 2 - Kayit", kare)
        tus = cv2.waitKey(1) & 0xFF
        zaman.tur("ekran")

        if tus == ord("q"):
            break
        if tus == ord("k"):
            kaydedici.degistir(kare)

    kaydedici.durdur()
    zaman.rapor()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




