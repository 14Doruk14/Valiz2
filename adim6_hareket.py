import cv2

HEDEF_MESAFE = 1.5   # m: sahibin bu kadar arkasında dur
V_MAX = 0.8          # m/s: en yüksek ileri hız
W_MAX = 1.0          # rad/s: en yüksek dönüş hızı
K_V = 0.6            # mesafe hatasının her metresi için ne kadar hız
K_W = 1.2            # yatay hatanın her birimi için ne kadar dönüş
OLU_BOLGE = 0.1      # bu kadar küçük yatay hatada dönme


def sinirla(deger, alt, ust):
    """Değeri [alt, ust] aralığına sıkıştırır."""
    return max(alt, min(ust, deger))


def hareket_karari(gorunuyor, hata, mesafe, kesik):
    """(V, W, durum yazısı) döndürür. V: m/s ileri, W: rad/s (+ sol, - sağ)."""
    if not gorunuyor or hata is None:
        return 0.0, 0.0, "KAYIP - DUR"

    # Dönüş
    if abs(hata) < OLU_BOLGE:
        w = 0.0
    else:
        w = sinirla(-K_W * hata, -W_MAX, W_MAX)

    # İleri
    if kesik or mesafe is None:
        v = 0.0
        durum = "COK YAKIN - DUR"
    else:
        v = sinirla(K_V * (mesafe - HEDEF_MESAFE), 0.0, V_MAX)
        durum = "TAKIP" if v > 0 else "YETERINCE YAKIN"

    # Keskin dönüşte yavaşla
    if abs(hata) > 0.5:
        v = v * 0.5

    return v, w, durum


def hareket_ciz(kare, v, w, durum):
    h, gen = kare.shape[:2]
    taban = (gen // 2, h - 20)
    uc = (int(gen // 2 - w * 100), int(h - 20 - v * 150))   # W>0 (sol) -> ok sola

    if durum == "TAKIP":
        renk = (0, 255, 0)
    elif durum == "KAYIP - DUR":
        renk = (0, 0, 255)
    else:
        renk = (0, 165, 255)

    if taban == uc:
        cv2.circle(kare, taban, 10, renk, 3)              # hareket yok: daire
    else:
        cv2.arrowedLine(kare, taban, uc, renk, 4, tipLength=0.3)

    cv2.putText(kare, f"V={v:.2f} m/s  W={w:+.2f} rad/s  {durum}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)


if __name__ == "__main__":
    print(hareket_karari(True, 0.0, 3.0, False))    # ortada, uzak   -> hızlı ileri, dönme yok
    print(hareket_karari(True, 0.0, 2.0, False))    # ortada, 2 m    -> yavaş ileri
    print(hareket_karari(True, 0.0, 1.2, False))    # ortada, yakın  -> dur
    print(hareket_karari(True, 0.4, 2.5, False))    # sağda          -> ileri + sağa (W eksi)
    print(hareket_karari(True, -0.7, 2.5, False))   # çok solda      -> yavaş ileri + sola (W artı)
    print(hareket_karari(True, 0.3, 2.3, True))     # kesik          -> dur ama dön
    print(hareket_karari(False, None, None, False)) # kayıp          -> dur

if __name__ == "__main__":
    print(hareket_karari(True, 0.0, 3.0, False))    # ortada, uzak   -> hızlı ileri, dönme yok
    print(hareket_karari(True, 0.0, 2.0, False))    # ortada, 2 m    -> yavaş ileri
    print(hareket_karari(True, 0.0, 1.2, False))    # ortada, yakın  -> dur
    print(hareket_karari(True, 0.4, 2.5, False))    # sağda          -> ileri + sağa (W eksi)
    print(hareket_karari(True, -0.7, 2.5, False))   # çok solda      -> yavaş ileri + sola (W artı)
    print(hareket_karari(True, 0.3, 2.3, True))     # kesik          -> dur ama dön
    print(hareket_karari(False, None, None, False)) # kayıp          -> dur