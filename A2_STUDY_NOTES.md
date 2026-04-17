# Assignment 2 – ArUco-Based Localization (Study Notes)

## Genel Amaç
Duckiebot'un **ArUco marker** tespiti ve **tekerlek odometri**sini birleştirerek haritadaki konumunu gerçek zamanlı olarak tahmin etmesi (localization).

---

## Sistem Mimarisi

```
Camera → aruco_detector → pose_estimator → (x, y, θ) düzeltmesi
Encoders → odometry      →                → (x, y, θ) dead-reckoning
                                          ↓
                                     visualizer → /visualization/compressed
```

---

## Modüller

### `config.py` – Sabitler
- **ArUco dict:** `DICT_APRILTAG_36h11`
- **Marker boyutu:** 6.75 cm (`MARKER_SIZE_M = 0.0675`)
- **TAG_POSES:** Haritadaki her tag'in dünya koordinatı `{id: (x_m, y_m, yaw_rad)}`
  - Örn: Tag 0 → (0.5, 0.0, 0°), Tag 3 → (0.5, 1.0, 180°)
- **Tekerlek parametreleri:** radius=0.0318 m, baseline=0.1 m, 135 tick/tur
- **Yayın hızı:** 30 Hz

---

### `calibration.py` – Kamera Kalibrasyonu
- İlk olarak YAML dosyasından yükler: `/data/config/calibrations/camera_intrinsic/<vehicle>.yaml`
- Bulamazsa `/camera_info` ROS topic'inden alır
- `K` (3×3 camera matrix) ve `D` (distortion coefficients) sağlar

---

### `odometry.py` – Tekerlek Odometrisi
- Sol/sağ encoder tick'lerini izler
- **`compute_delta()`** her çağrıda `(d_center, d_theta)` döndürür:
  ```
  dl = Δleft_ticks / 135 * 2π * 0.0318
  dr = Δright_ticks / 135 * 2π * 0.0318
  d_center = (dl + dr) / 2
  d_theta  = (dr - dl) / 0.1
  ```
- **`reset_to_current()`**: ArUco düzeltmesinden sonra çağrılır → accumulated error sıfırlanır

---

### `aruco_detector.py` – Marker Tespiti
- OpenCV 4.6 öncesi ve sonrası API farklarını handle eder
- **`detect_and_annotate(img, K)`**:
  1. `detectMarkers` ile köşeleri bulur
  2. Her marker için `solvePnP(SOLVEPNP_IPPE_SQUARE)` ile `rvec, tvec` hesaplar
  3. En yakın tag'i döndürür: `(tag_id, rvec, tvec)`
- `tvec` = kamera çerçevesinde marker'ın pozisyonu (metre)
- `rvec` = Rodrigues rotation vector

---

### `pose_estimator.py` – Dünya Koordinatına Çevirme
**`from_tag(tag_id, rvec, tvec)`** → `(x, y, θ)` world frame

Adımlar:
1. `R_cm = Rodrigues(rvec)` → marker→camera rotasyon matrisi
2. `cam_in_marker = -R_cm^T · tvec` → kameranın marker frame'indeki pozisyonu
3. `yaw_cam_to_marker = atan2(-R_cm[2,0], R_cm[2,2])` → heading açısı
4. Tag'in bilinen world pose'u ile döndür → robot'un dünya koordinatı

**`tag_from_robot()`** → tersine: robot pozisyonundan tag'in dünya koordinatını kayıt eder (ilk kez görülen tag'ler için)

---

### `localization_node.py` – Ana Node
**Subscribe ettiği topic'ler:**
- `/mouse/camera_node/image/compressed` – kamera görüntüsü
- `/mouse/camera_node/camera_info` – kamera intrinsic'leri
- `/mouse/left_wheel_encoder_node/tick`
- `/mouse/right_wheel_encoder_node/tick`

**Publish ettiği topic:**
- `/mouse/assignment2/visualization/compressed`

**`_image_cb` akışı (her frame):**
1. Encoder'lardan odometri güncelle → `(x, y, θ)` dead-reckoning ile ilerlет
2. Kalibrasyon hazır değilse dur
3. `getOptimalNewCameraMatrix` + `undistort` → lens distorsiyonu gider
4. ArUco detection → `(tag_id, rvec, tvec)`
5. **Tag varsa:** `PoseEstimator.from_tag()` → pose'u düzelt, odometriyi resetle
6. **Tag yoksa:** sadece odometri kullan

---

### `visualizer.py` – Görselleştirme
İki panel üst üste yayınlar:
- **Üst (400×300):** Kamera görüntüsü + ArUco overlay'leri
- **Alt (400×300):** Top-down harita
  - Grid: 0.5 m aralıklı
  - Siyah kareler: TAG_POSES'daki marker'lar
  - **Yeşil** nokta/ok: ArUco ile düzeltilmiş pose
  - **Turuncu** nokta/ok: Sadece odometri

---

## Localization Logic Özeti

```
Her frame:
  (x,y,θ) += odometry_delta    # dead-reckoning

  if ArUco tag görünüyorsa:
    (x,y,θ) = PoseEstimator.from_tag(...)  # ground truth'a snap
    odometry.reset_to_current()             # drift'i sıfırla
    source = "aruco"
  else:
    source = "odometry"
```

**Neden hybrid?** Sadece odometri → drift birikir. Sadece ArUco → tag görünmeyince çalışmaz. İkisi birlikte → ArUco her görüldüğünde hatayı düzeltir, arada odometri devam ettirir.

---

## Olası Sorular & Cevaplar

**Q: ArUco marker nedir?**  
A: Kare binary pattern'li fiducial marker. Her ID için benzersiz desen. `solvePnP` ile kameraya göre tam 6DoF pose (pozisyon + rotasyon) ölçülebilir.

**Q: rvec ve tvec ne anlama gelir?**  
A: `tvec` = kamera merkezinden marker merkezine uzaklık vektörü (metre). `rvec` = Rodrigues formatında rotasyon vektörü (yönü = rotasyon ekseni, büyüklüğü = açı rad).

**Q: Odometri neden yeterli değil?**  
A: Tekerlek kayması, zemin düzensizlikleri, encoder hassasiyetsizliği zamanla hata biriktirir (drift).

**Q: SOLVEPNP_IPPE_SQUARE neden kullanıldı?**  
A: Kare marker'lar için özel optimized yöntem. 4 köşeli kare problemini daha hızlı ve kararlı çözer (standart iteratif'ten daha iyi).

**Q: `reset_to_current()` neden gerekli?**  
A: ArUco düzeltmesi yapıldıktan sonra odometri bir önceki (hatalı) pozisyona göre değil, düzeltilmiş pozisyona göre devam etmeli. Yoksa `compute_delta()` bir sonraki çağrıda hatalı Δ hesaplar.

**Q: TAG_POSES'da olmayan bir tag görülürse ne olur?**  
A: `tag_from_robot()` ile o anki robot pose'undan tersine hesaplanıp `TAG_POSES`'a eklenir (runtime registration).

**Q: Kamera kalibrasyonu olmadan çalışır mı?**  
A: Odometri devam eder ama ArUco detection yapılmaz (`if not self._calib.is_ready: return`).

**Q: Publish hızı kaç?**  
A: 30 Hz (görselleştirme timer'ı). Image callback ise kamera frame hızında çalışır.

**Q: `alpha=1` ne anlama gelir `getOptimalNewCameraMatrix`'te?**  
A: Tüm orijinal piksellar korunur (siyah kenar boşlukları oluşabilir). `alpha=0` olsaydı siyah alanlar kırpılırdı.

---

## Koordinat Sistemi

```
Marker frame:  x=sağ, y=yukarı, z=dışarı (tag yüzeyi)
Camera frame:  x=sağ, y=aşağı, z=ileri (sahneye doğru)
World frame:   x=ileri, y=sol, z=yukarı (standart ROS 2D map)
```
