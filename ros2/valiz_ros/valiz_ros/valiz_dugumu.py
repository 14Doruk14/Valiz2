"""Valiz düğümü: /image'den kareleri alır, sahibini takip eder, /cmd_vel yayınlar."""
import cv2
import os
import sys
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


VALIZ_KLASORU = "/mnt/c/Users/USER/Desktop/Valiz2"
sys.path.insert(0, VALIZ_KLASORU)          # valiz.py ve adim*.py buradan bulunsun

PENCERE = "Valiz 2 - ROS"

from ultralytics import YOLO
import valiz
from adim2_tespit import kisileri_bul
from adim3_takip import Takipci, cift_kutulari_ayikla
from adim4_sahip import tiklanani_bul, fare
from adim5_konum import konum_ciz
from adim6_hareket import hareket_karari, hareket_ciz
from adim7b_osnet import osnet_yukle
from adim8_olcum import Zamanlayici, zaman_ciz

class ValizDugumu(Node):

    def __init__(self):
        super().__init__("valiz_dugumu")
        self.kopru = CvBridge()

        os.chdir(VALIZ_KLASORU)            # yolo11n.pt ve modeller/ buradan bulunsun
        valiz.ayarlari_uygula()
        self.model = YOLO("yolo11n.pt")
        self.takipci = Takipci()
        self.sahip = valiz.SahipTakibi(osnet_yukle(valiz.OSNET_MODELI))
        self.konum = valiz.SahipKonumu()
        self.onceki_durum = None

        self.zaman = Zamanlayici()
        self.cikis = False

        self.declare_parameter("ekransiz", False)
        self.ekransiz = self.get_parameter("ekransiz").value
        self.tik = {"tik": None}
        if not self.ekransiz:
            cv2.namedWindow(PENCERE)
            cv2.setMouseCallback(PENCERE, fare, self.tik)

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.hiz_yayinci = self.create_publisher(Twist, "/cmd_vel", 10)
        self.abone = self.create_subscription(Image, "/image", self.kare_geldi, qos)
        self.get_logger().info("/image dinleniyor")

    def kare_geldi(self, mesaj):
        """/image'e her kare geldiğinde ROS bu fonksiyonu çağırır."""
        self.zaman.basla()
        kare = self.kopru.imgmsg_to_cv2(mesaj, desired_encoding="bgr8")
        kare_yuk, kare_gen = kare.shape[:2]
        self.zaman.tur("donusum")

        kisiler = cift_kutulari_ayikla(kisileri_bul(self.model, kare))
        self.zaman.tur("yolo")
        gorunenler = self.takipci.guncelle(kisiler)
        self.zaman.tur("takip")

        if self.tik["tik"] is not None:
            secilen = tiklanani_bul(gorunenler, self.tik["tik"])
            if secilen is not None:
                self.sahip.sec(secilen)
            self.tik["tik"] = None
        self.sahip.guncelle(kare, gorunenler)
        self.zaman.tur("sahip")

        gorunuyor = self.konum.guncelle(self.sahip.sahip_id, gorunenler, kare_gen, kare_yuk)
        v, w, durum = hareket_karari(gorunuyor, self.konum.hata, self.konum.mesafe, self.konum.kesik)

        komut = Twist()
        komut.linear.x = float(v)
        komut.angular.z = float(w)
        self.hiz_yayinci.publish(komut)
        self.zaman.tur("hareket")

        if durum != self.onceki_durum:
            self.get_logger().info(f"Durum: {durum}   V={v:.2f}  W={w:+.2f}")
            self.onceki_durum = durum

        if self.ekransiz:
            return

        kare[:valiz.YAZI_SERIDI] //= 2
        self.sahip.ciz(kare, gorunenler)
        if gorunuyor:
            konum_ciz(kare, gorunenler[self.sahip.sahip_id], self.konum.hata, self.konum.mesafe, self.konum.kesik)
        hareket_ciz(kare, v, w, durum)
        zaman_ciz(kare, self.zaman)

        cv2.imshow(PENCERE, kare)
        self.zaman.tur("ekran")

    def tuslari_oku(self):
        """Her döngü turunda çağrılır; kare gelmese de tuşlar okunur."""
        if self.ekransiz:
            return
        tus = cv2.waitKey(1) & 0xFF
        if tus == ord("q"):
            self.cikis = True
        if tus == ord("r"):
            self.sahip.birak()
            self.get_logger().info("Sahip bırakıldı")

def main():
    rclpy.init()
    dugum = ValizDugumu()
    try:
        while rclpy.ok() and not dugum.cikis:
            rclpy.spin_once(dugum, timeout_sec=0.1)
            dugum.tuslari_oku()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        dugum.destroy_node()
        rclpy.try_shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


















