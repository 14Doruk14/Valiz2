"""Bir video dosyasını /image konusuna yayınlar. Gerçek kameranın yerine geçer."""
import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

VARSAYILAN_VIDEO = "/mnt/c/Users/USER/Desktop/Valiz2/kayitlar/video_01.mp4"


class VideoYayinci(Node):
    def __init__(self):
        super().__init__("video_yayinci")
        self.declare_parameter("video", VARSAYILAN_VIDEO)
        self.declare_parameter("basa_sar", True)
        yol = self.get_parameter("video").value
        self.basa_sar = self.get_parameter("basa_sar").value

        self.cap = cv2.VideoCapture(yol)
        if not self.cap.isOpened():
            raise RuntimeError(f"Vido açılamadı: {yol}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.kopru = CvBridge()
        self.yayinci = self.create_publisher(Image, "/image", 10)
        self.zamanlayici = self.create_timer(1.0 / fps, self.kare_yayinla)
        self.get_logger().info(f"{yol} yayınlanıyor ({fps:.0f} FPS) -> /image")

    def kare_yayinla(self):
        """Zamanlayıcı her tetiklendiğinde bir kare okur ve yayınlar."""
        ok , kare = self.cap.read()
        if not ok:
            if not self.basa_sar:
                self.get_logger().info("Video bitti.")
                self.zamanlayici.cancel()
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # başa sar
            ok , kare = self.cap.read()
            if not ok:
                return
        mesaj = self.kopru.cv2_to_imgmsg(kare, encoding="bgr8")
        mesaj.header.stamp = self.get_clock().now().to_msg()
        mesaj.header.frame_id = "kamera"
        self.yayinci.publish(mesaj)
        
        
def main():
    rclpy.init()
    dugum = VideoYayinci()
    try:
        rclpy.spin(dugum)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        dugum.cap.release()
        dugum.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()











