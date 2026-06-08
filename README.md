# Card Detective — Real-Time Card Recognition Game with OpenCV

> **Proyek Computer Vision (OpenCV)**
> **Departemen Teknik Komputer, Institut Teknologi Sepuluh Nopember (ITS)**

---

## Demo & Overview

**Card Detective** adalah game interaktif berbasis **Computer Vision** yang menggunakan kamera secara *real-time* untuk mendeteksi dan mengenali kartu fisik. Pemain ditantang untuk menemukan dan mengarahkan kartu target ke kamera sebelum waktu habis — sistem secara otomatis akan mengenali kartu mana yang sedang dipegang.

Seluruh pipeline deteksi dibangun **dari nol menggunakan OpenCV murni**, tanpa library machine learning eksternal seperti TensorFlow atau PyTorch. Teknik yang digunakan mencakup *edge detection*, *contour analysis*, *perspective transform*, dan *pixel-level image matching*.

---

## Teknik Computer Vision yang Diimplementasikan

Ini adalah inti dari proyek ini. Setiap frame kamera diproses melalui sebuah pipeline multi-tahap:

### Pipeline Deteksi Kartu (Per Frame)

```
[Frame BGR dari Kamera]
        │
        ▼
[Grayscale Conversion]  →  cv2.cvtColor(frame, COLOR_BGR2GRAY)
        │
        ▼
[Gaussian Blur]         →  cv2.GaussianBlur(gray, (5,5), 0)
        │                   Mengurangi noise sebelum edge detection
        ▼
[Canny Edge Detection]  →  cv2.Canny(blur, 50, 150)
        │                   Mendeteksi tepi objek berdasarkan gradien intensitas piksel
        ▼
[Contour Extraction]    →  cv2.findContours(..., RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        │                   Mengekstrak outline tertutup dari hasil Canny
        ▼
[Polygon Approximation] →  cv2.approxPolyDP(c, 0.02 * peri, True)
        │                   Menyederhanakan kontur menjadi polygon — filter 4 titik = kartu
        ▼
[Perspective Transform] →  cv2.getPerspectiveTransform + cv2.warpPerspective
        │                   Meluruskan sudut pandang miring menjadi gambar frontal 200×300px
        ▼
[Pixel Difference Matching] → np.mean(cv2.absdiff(warped, db_img))
                            Mencocokkan kartu dengan database menggunakan MAE (Mean Absolute Error)
```

### 1. Canny Edge Detection

Canny digunakan sebagai tahap pertama deteksi kartu. Dengan threshold `50–150`, algoritma mendeteksi perubahan gradien intensitas piksel yang merepresentasikan tepi kartu fisik, bahkan dalam kondisi pencahayaan yang bervariasi.

```python
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blur, 50, 150)
```

Gaussian Blur dipasang sebelum Canny untuk menekan noise sensor kamera yang dapat memicu false edge.

### 2. Contour Analysis & Quadrilateral Filter

Dari semua kontur yang ditemukan, sistem hanya memproses kontur yang:
- Memiliki tepat **4 titik sudut** setelah `approxPolyDP` (bentuk segiempat).
- Memiliki **luas area > 1000 piksel** untuk mengabaikan noise kecil.

```python
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4 and cv2.contourArea(c) > 1000:
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
```

Pendekatan ini jauh lebih ringan dibanding deep learning untuk kasus deteksi objek berbentuk segiempat, karena tidak membutuhkan proses training maupun GPU.

### 3. Four-Point Perspective Transform

Kartu yang ditangkap kamera hampir tidak pernah frontal sempurna — selalu ada distorsi perspektif akibat sudut pengambilan gambar. Fungsi `four_point_transform()` mengoreksi distorsi ini menggunakan transformasi perspektif homografi:

```python
def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]   # Sudut kiri-atas & kanan-bawah
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)] # Sudut kanan-atas & kiri-bawah
    dst = np.array([[0, 0], [199, 0], [199, 299], [0, 299]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (200, 300))
```

Hasilnya adalah gambar kartu yang selalu berukuran **200×300 piksel** dan tegak lurus, apapun sudut kamera saat mengambil gambar.

### 4. Pixel-Level Image Matching (Mean Absolute Error)

Kartu yang sudah di-*warp* kemudian dicocokkan ke seluruh database menggunakan metrik **Mean Absolute Error (MAE)** per piksel:

```python
def match_card(warped_img, database):
    best_match = None
    min_diff = float('inf')
    for name, db_img in database.items():
        score = np.mean(cv2.absdiff(warped_img, db_img))
        if score < min_diff:
            min_diff = score
            best_match = name
    return best_match if min_diff < 75 else None
```

- `cv2.absdiff()` menghitung selisih absolut setiap piksel antara gambar kartu yang ditangkap dan gambar referensi di database.
- Threshold `< 75` memastikan sistem tidak memberikan hasil positif palsu ketika kartu yang dipegang tidak ada dalam database.

---

## Fitur Game

- **State Machine** dengan 4 state: `MENU → PLAYING → FEEDBACK → GAMEOVER`, mengatur alur game secara bersih tanpa blocking.
- **Countdown Timer 10 detik** dengan timer bar visual yang berubah warna (hijau → merah saat kritis).
- **Sistem Nyawa 3 lives** — berkurang saat waktu habis tanpa menemukan kartu yang tepat.
- **Auto-progression** — setelah menampilkan feedback 1.5 detik, sistem langsung reset dan memilih kartu target baru secara otomatis.
- **Live scan preview** — area kecil di sidebar menampilkan hasil *warp* kartu yang sedang terdeteksi secara real-time.
- **Custom UI layout** yang dirender langsung di atas `numpy array` menggunakan `cv2.rectangle`, `cv2.putText`, dan `cv2.line` — tanpa framework GUI eksternal.

---

## Struktur Proyek

```
.
├── Search_Card_Game.py     → Source code utama game
└── dataset_kartu/          → Folder database kartu referensi
    ├── nama_kartu_1.jpg
    ├── nama_kartu_2.jpg
    └── ...
```

Setiap gambar di `dataset_kartu/` adalah foto kartu fisik yang sudah diambil secara frontal, di-*resize* ke **200×300 piksel** dalam format grayscale, dan digunakan sebagai template referensi untuk pencocokan.

---

## Cara Menambah Kartu ke Dataset

1. Foto kartu fisik dengan pencahayaan merata dan latar belakang kontras.
2. Crop dan resize gambar ke **200 × 300 piksel** dalam format grayscale.
3. Simpan sebagai `nama_kartu.jpg` di dalam folder `dataset_kartu/`.
4. Nama file (tanpa ekstensi) otomatis menjadi nama kartu yang ditampilkan di game.

---

## Instalasi & Menjalankan

### Prasyarat

```bash
pip install opencv-python numpy
```

### Jalankan

```bash
python Search_Card_Game.py
```

Pastikan folder `dataset_kartu/` sudah berisi minimal satu file `.jpg` dan kamera terhubung sebelum menjalankan.

### Kontrol Keyboard

| Tombol | Fungsi |
|---|---|
| `Enter` | Mulai game dari menu |
| `R` | Restart setelah Game Over |
| `Q` | Keluar dari aplikasi |

---

## Spesifikasi Teknis

| Komponen | Detail |
|---|---|
| **Bahasa** | Python 3.x |
| **Library Utama** | OpenCV (`cv2`), NumPy |
| **Input** | Webcam (default device index 0) |
| **Resolusi Canvas** | 1000 × 600 piksel |
| **Resolusi Kamera** | 640 × 480 piksel |
| **Resolusi Template** | 200 × 300 piksel (grayscale) |
| **Metode Deteksi** | Canny Edge + Contour Analysis + Perspective Transform |
| **Metode Pencocokan** | Pixel MAE dengan threshold 75 |
| **FPS Target** | Real-time (~30 FPS, `cv2.waitKey(1)`) |

---

## Pengembang

| | |
|---|---|
| **Nama** | Ahmad Dafa Salam |
| **NRP** | 5024231024 |
| **Program Studi** | S1 Teknik Komputer |
| **Institusi** | Institut Teknologi Sepuluh Nopember (ITS) |
