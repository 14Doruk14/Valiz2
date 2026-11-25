import sys 
import cv2

def kaynak_ac(kaynak):
    """Kaynak sayıysa webcam'i, değilse video dosyasını açar."""
    if kaynak.isdigit():
        cap = cv2.VideoCapture(int(kaynak), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(kaynak)

    if not cap.isOpened():
        print("Kaynak açılmadı:", kaynak)
        sys.exit(1)
    return cap

def main():
    # Komut satırında bir şey verilmezse 0 numaralı webcam
    if len(sys.argv) > 1:
        kaynak = sys.argv[1]
    else:
        kaynak = "0"

    cap = kaynak_ac(kaynak)

    ilk_kare = True
    while True:
        ok, kare = cap.read()
        if not ok:
            print("Kare gelmedi (video bitti ya da kamera koptu).")
            break

        if ilk_kare:
            print("Kare boyutu (yükseklik, genişlik, kanal):", kare.shape)
            ilk_kare = False

        cv2.imshow("Valiz 2", kare)

        tus = cv2.waitKey(1) & 0xFF
        if tus == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()












