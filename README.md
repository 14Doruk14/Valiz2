# Valiz 2

Sahibini takip eden valiz robotu — adım adım, sıfırdan.

| Dosya | Ne yapıyor |
|---|---|
| adim1_kamera.py | Kamera / video açma |
| adim2_tespit.py | YOLO11n ile kişi tespiti |
| adim3_takip.py | IoU ile takip, kalıcı ID'ler |
| adim4_sahip.py | Sahibi seçme (otomatik / tıklama) |
| adim5_konum.py | Yatay hata ve mesafe tahmini |
| adim6_hareket.py | V / W hareket kararı |
| adim7_reid.py | Yeniden tanıma: renk histogramı (7a) |
| adim7b_osnet.py | Yeniden tanıma: OSNet (7b, seçilen) |
| adim8_olcum.py | Süre / FPS ölçümü ve video kaydı |

Gerekenler (depoda yok): `yolo11n.pt` (ultralytics otomatik indirir), `modeller/osnet_x0_25_msmt17.onnx`.
