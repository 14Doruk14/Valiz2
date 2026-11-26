"""Valiz 2 - ana program: sahibini bulur, takip eder, V/W hareket komutu üretir."""
import argparse
import cv2
from ultralytics import YOLO

import adim2_tespit
import adim3_takip
import adim4_sahip
import adim5_konum
import adim6_hareket
import adim7b_osnet
from adim1_kamera import kaynak_ac
from adim2_tespit import kisileri_bul
from adim3_takip import Takipci, cift_kutulari_ayikla
from adim4_sahip import tiklanani_bul, fare, ciz as sahip_ciz
from adim5_konum import yatay_hata, kesik_mi, mesafe_tahmin, yumusat, konum_ciz
from adim6_hareket import hareket_karari, hareket_ciz
from adim7b_osnet import (osnet_yukle, gorunum_cikar, galeriye_benzerlik,
                          baska_kisiyle_ortusuyor, reid_ciz)
from adim8_olcum import Zamanlayici, zaman_ciz, Kaydedici


# ======================= AYARLAR =======================
# Bütün ayarlanabilir sayılar burada. Hangi dosyanın ayarı olduğu solda yazıyor.
AYARLAR = {
    adim2_tespit: {
        "GUVEN_ESIGI": 0.25,      # YOLO bundan emin değilse kişiyi atar
    },
    adim3_takip: {
        "IOU_ESIGI": 0.3,         # bu kadar örtüşmüyorsa farklı kişi
        "MAX_KAYIP": 15,          # kaç kare görülmezse iz silinir
        "YENI_ID_ESIGI": 0.5,     # yeni ID açmak için gereken güven
    },
    adim4_sahip: {
        "MIN_ALAN_ORANI": 0.05,   # karenin bu oranından küçük kişi otomatik seçilmez
    },
    adim5_konum: {
        "KISI_BOYU_M": 1.70,      # mesafe tahmini için ortalama boy
        "YUMUSATMA": 0.3,         # hata/mesafe yumuşatma (0-1, büyük = hızlı tepki)
    },
    adim6_hareket: {
        "HEDEF_MESAFE": 1.5,      # sahibin arkasında durulacak mesafe (m)
        "V_MAX": 0.8,             # en yüksek ileri hız (m/s)
        "W_MAX": 1.0,             # en yüksek dönüş hızı (rad/s)
        "K_V": 0.6,               # mesafe hatası -> hız katsayısı
        "K_W": 1.2,               # yatay hata -> dönüş katsayısı
        "OLU_BOLGE": 0.1,         # bu kadar küçük yatay hatada dönme
    },
    adim7b_osnet: {
        "BENZERLIK_ESIGI": 0.7,   # OSNet kosinüs benzerliği eşiği
        "ONAY_KARE": 5,           # yeniden tanıma için art arda kaç kare
        "GALERI_BOYUTU": 10,      # sahibin kaç görünüşü saklanır
        "GALERI_ARALIK": 10,      # kaç karede bir galeriye örnek eklenir
    },
}
PENCERE = "Valiz 2"
OSNET_MODELI = "modeller/osnet_x0_25_msmt17.onnx"
YAZI_SERIDI = 160     # ekranın üstünde bilgi yazılarının olduğu şerit (piksel)
# =======================================================


class SahipTakibi:
    """Sahibin kim olduğunu tutar: seçer, görünüşünü hatırlar, kaybolursa geri bulur."""

    def __init__(self, oturum):
        self.oturum = oturum          # OSNet modeli
        self.sahip_id = None
        self.galeri = []              # sahibin son görünüşleri (OSNet vektörleri)
        self.kare_no = 0
        self.aday_id = None           # "bu sahip olabilir" denen kişi
        self.aday_sayac = 0           # aday kaç karedir art arda benziyor
        self.benzerlikler = {}        # sadece kayıpken dolar: {iz_id: benzerlik}

    def sec(self, iz_id):
        """Yeni sahip: eski galeri başkasına ait, sıfırla."""
        self.sahip_id = iz_id
        self.galeri = []
        self.aday_id, self.aday_sayac = None, 0

    def birak(self):
        self.sec(None)

    def gorunuyor_mu(self, gorunenler):
        return self.sahip_id is not None and self.sahip_id in gorunenler

    def guncelle(self, kare, gorunenler):
        """Her karede bir kez çağrılır. Sahibin (güncel) ID'sini döndürür."""
        self.kare_no += 1
        self.benzerlikler = {}

        if self.sahip_id is None:
            h, w = kare.shape[:2]
            yeni = adim4_sahip.otomatik_sec(gorunenler, w, h)
            if yeni is not None:
                self.sec(yeni)
        elif self.sahip_id in gorunenler:
            self._galeriye_ekle(kare, gorunenler)
        else:
            self._yeniden_bul(kare, gorunenler)
        return self.sahip_id

    def _galeriye_ekle(self, kare, gorunenler):
        """Sahip görünüyor: arada bir görünüşünü galeriye kaydet."""
        self.aday_id, self.aday_sayac = None, 0
        kutu = gorunenler[self.sahip_id]
        zamani_geldi = not self.galeri or self.kare_no % adim7b_osnet.GALERI_ARALIK == 0
        if not zamani_geldi or baska_kisiyle_ortusuyor(kutu, gorunenler, self.sahip_id):
            return
        oz = gorunum_cikar(self.oturum, kare, kutu)
        if oz is not None:
            self.galeri.append(oz)
            if len(self.galeri) > adim7b_osnet.GALERI_BOYUTU:
                self.galeri.pop(0)

    def _yeniden_bul(self, kare, gorunenler):
        """Sahip kayıp: herkesi galeriyle karşılaştır, art arda benzeyeni sahip yap."""
        if not self.galeri:
            return
        for iz_id, kutu in gorunenler.items():
            oz = gorunum_cikar(self.oturum, kare, kutu)
            self.benzerlikler[iz_id] = galeriye_benzerlik(self.galeri, oz)

        if not self.benzerlikler:
            self.aday_id, self.aday_sayac = None, 0
            return
        en_iyi = max(self.benzerlikler, key=self.benzerlikler.get)
        if self.benzerlikler[en_iyi] < adim7b_osnet.BENZERLIK_ESIGI:
            self.aday_id, self.aday_sayac = None, 0
            return

        if en_iyi == self.aday_id:
            self.aday_sayac += 1
        else:
            self.aday_id, self.aday_sayac = en_iyi, 1

        if self.aday_sayac >= adim7b_osnet.ONAY_KARE:
            print(f"Sahip geri geldi: ID {self.sahip_id} -> ID {en_iyi}")
            self.sahip_id = en_iyi          # galeri korunur: aynı kişi
            self.aday_id, self.aday_sayac = None, 0

    def ciz(self, kare, gorunenler):
        sahip_ciz(kare, gorunenler, self.sahip_id, YAZI_SERIDI)
        reid_ciz(kare, gorunenler, self.benzerlikler, len(self.galeri),
                 self.aday_id, self.aday_sayac)


class SahipKonumu:
    """Sahibin yumuşatılmış yatay hatası ve mesafesi. Sahip değişince sıfırlanır."""

    def __init__(self):
        self.iz_id = None             # değerler kimin için hesaplandı
        self.hata = None              # -1 (sol) ... +1 (sağ)
        self.mesafe = None            # metre
        self.kesik = False            # kutu üst/alt kenara değiyor mu

    def guncelle(self, sahip_id, gorunenler, kare_gen, kare_yuk):
        """Sahip görünüyorsa değerleri günceller ve True döndürür."""
        if sahip_id != self.iz_id:
            self.iz_id = sahip_id
            self.hata = None
            self.mesafe = None
        if sahip_id not in gorunenler:
            return False

        kutu = gorunenler[sahip_id]
        self.hata = yumusat(self.hata, yatay_hata(kutu, kare_gen))
        m = mesafe_tahmin(kutu, float(kare_gen))
        if m is not None:
            self.mesafe = yumusat(self.mesafe, m)
        self.kesik = kesik_mi(kutu, kare_yuk)
        return True


def komut_gonder(v, w):
    """V/W'nin programdan çıktığı TEK yer.
    Şimdilik bir şey yapmıyor. İleride: Adım 10'da ESP32'ye seri port, ROS'ta /cmd_vel."""
    pass


def ayarlari_uygula():
    """AYARLAR'daki değerleri ilgili dosyaların sabitlerinin üstüne yazar."""
    for modul, degerler in AYARLAR.items():
        for ad, deger in degerler.items():
            if not hasattr(modul, ad):
                raise ValueError(f"{modul.__name__} içinde {ad} diye bir ayar yok (yazım hatası?)")
            setattr(modul, ad, deger)


def secenekleri_oku():
    p = argparse.ArgumentParser(description="Valiz 2 - sahibini takip eden valiz")
    p.add_argument("kaynak", nargs="?", default="0",
                   help="kamera numarası (0, 1 ...) ya da video dosyası")
    p.add_argument("--ekransiz", action="store_true",
                   help="pencere açma (Pi'de ekran yokken)")
    p.add_argument("--kayit", action="store_true",
                   help="işlenmiş görüntüyü kayitlar/ klasörüne kaydet")
    return p.parse_args()


def main():
    sec = secenekleri_oku()
    ayarlari_uygula()
    print(f"Kaynak: {sec.kaynak}   ekransız: {sec.ekransiz}   kayıt: {sec.kayit}")

    cap = kaynak_ac(sec.kaynak)
    model = YOLO("yolo11n.pt")
    takipci = Takipci()
    sahip = SahipTakibi(osnet_yukle(OSNET_MODELI))
    konum = SahipKonumu()
    onceki_durum = None
    zaman = Zamanlayici()
    kaydedici = Kaydedici() if sec.kayit else None
    cizim_gerekli = not sec.ekransiz or sec.kayit

    durum = {"tik": None}
    if not sec.ekransiz:
        cv2.namedWindow(PENCERE)
        cv2.setMouseCallback(PENCERE, fare, durum)

    try:
        while True:
            zaman.basla()
            ok, kare = cap.read()
            if not ok:
                break
            zaman.tur("kamera")
            kare_yuk, kare_gen = kare.shape[:2]

            kisiler = cift_kutulari_ayikla(kisileri_bul(model, kare))
            zaman.tur("yolo")
            gorunenler = takipci.guncelle(kisiler)
            zaman.tur("takip")

            if durum["tik"] is not None:
                secilen = tiklanani_bul(gorunenler, durum["tik"])
                if secilen is not None:
                    sahip.sec(secilen)
                durum["tik"] = None
            sahip.guncelle(kare, gorunenler)
            zaman.tur("sahip")

            gorunuyor = konum.guncelle(sahip.sahip_id, gorunenler, kare_gen, kare_yuk)
            v, w, hareket_durumu = hareket_karari(gorunuyor, konum.hata, konum.mesafe, konum.kesik)
            komut_gonder(v, w)
            zaman.tur("hareket")

            if sec.ekransiz and hareket_durumu != onceki_durum:
                print(f"Durum: {hareket_durumu}")
                onceki_durum = hareket_durumu

            if cizim_gerekli:
                kare[:YAZI_SERIDI] //= 2          # üst şeridi karart: yazılar okunsun
                sahip.ciz(kare, gorunenler)
                if gorunuyor:
                    konum_ciz(kare, gorunenler[sahip.sahip_id], konum.hata, konum.mesafe, konum.kesik)
                hareket_ciz(kare, v, w, hareket_durumu)
                zaman_ciz(kare, zaman)

            if kaydedici is not None:
                if not kaydedici.kayitta_mi():
                    kaydedici.degistir(kare)      # ilk karede kaydı başlat
                kaydedici.yaz(kare)

            if not sec.ekransiz:
                cv2.imshow(PENCERE, kare)
                tus = cv2.waitKey(1) & 0xFF
                if tus == ord("q"):
                    break
                if tus == ord("r"):
                    sahip.birak()
            zaman.tur("ekran")

    except KeyboardInterrupt:
        print("\nCtrl+C ile durduruldu.")

    finally:
        if kaydedici is not None:
            kaydedici.durdur()
        zaman.rapor()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()