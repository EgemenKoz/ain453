#!/usr/bin/env python3

"""
ArUco marker görsellerini oluşturur.
Çalıştır:  python3 generate_markers.py
Çıktı:     markers/ klasörüne PNG dosyaları kaydeder.
Bunları kağıda yazdırıp haritaya yerleştirin.
"""

import os
import cv2
import cv2.aruco as aruco

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3, 4, 5]       # oluşturulacak tag ID'leri
MARKER_SIZE_PX = 400                    # piksel cinsinden görsel boyutu
MARKER_SIDE_CM = 6.5                    # fiziksel boyut (yazdırırken bu ölçüye ayarlayın)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "markers")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    aruco_dict = aruco.Dictionary_get(ARUCO_DICT_TYPE)

    for mid in MARKER_IDS:
        img = aruco.drawMarker(aruco_dict, mid, MARKER_SIZE_PX)
        # Beyaz kenarlık ekle (tespit doğruluğu artar)
        bordered = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        path = os.path.join(OUTPUT_DIR, f"aruco_4x4_id{mid}.png")
        cv2.imwrite(path, bordered)
        print(f"Saved: {path}  (ID={mid}, print at {MARKER_SIDE_CM} cm)")

    print(f"\n{len(MARKER_IDS)} marker saved to {OUTPUT_DIR}/")
    print(f"Print each marker at exactly {MARKER_SIDE_CM} cm x {MARKER_SIDE_CM} cm.")


if __name__ == "__main__":
    main()
