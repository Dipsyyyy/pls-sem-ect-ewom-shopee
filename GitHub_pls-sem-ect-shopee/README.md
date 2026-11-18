# Analisis PLS-SEM: Rantai Expectation Confirmation Theory Menuju eWOM dan Reputasi Online Shopee

Repositori ini berisi kode dan data untuk artikel:

> [[Nama Penulis]]. *Peran Mediasi Kepuasan dalam Rantai Expectation Confirmation Theory Menuju eWOM dan Reputasi Online Shopee*. Building of Informatics, Technology and Science (BITS), [[tahun, volume, nomor, halaman, DOI]].

Penelitian ini menguji model ekspektasi (EXP), konfirmasi (CONF), kepuasan (SAT), electronic word-of-mouth (EW), dan reputasi online (REP) pada 190 pengguna Shopee di Indonesia dengan Partial Least Squares Structural Equation Modeling (PLS-SEM).

## Isi repositori

```
kode/
  Analisis_PLS-SEM_Shopee.py        Analisis utama (Python, untuk Google Colab)
  Verifikasi_seminr_Shopee.ipynb    Verifikasi hasil dengan paket R seminr
data/
  smartpls_A_n190.csv               Skor 15 butir dari 190 responden (dataset A)
```

## Model penelitian

| Hipotesis | Jalur |
|---|---|
| H1 | EXP → CONF |
| H2 | EXP → SAT |
| H3 | CONF → SAT |
| H4 | CONF → EW |
| H5 | SAT → EW |
| H6 | EW → REP |

Kelima konstruk bersifat reflektif dan masing-masing diukur dengan tiga butir berskala Likert 1 (sangat tidak setuju) sampai 5 (sangat setuju).

## Kamus data (`data/smartpls_A_n190.csv`)

| Kolom | Isi |
|---|---|
| `ID` | Nomor urut responden pada data mentah |
| `EXP1`–`EXP3` | Ekspektasi |
| `CONF1`–`CONF3` | Konfirmasi |
| `SAT1`–`SAT3` | Kepuasan |
| `EW1`–`EW3` | eWOM (kesediaan membagikan ulasan) |
| `REP1`–`REP3` | Reputasi online |

Dataset A berisi responden yang berbelanja di Shopee dalam tiga bulan terakhir (7 dari 197 respons dikeluarkan). Sebanyak 35 responden memberi jawaban seragam pada seluruh butir dan tetap dimasukkan dalam analisis utama.

## Cara menjalankan

### 1. Analisis utama (Python)

Skrip ini dimulai dari data mentah ekspor Google Form (197 baris, 20 kolom), lalu melakukan audit data, membentuk dataset A, menghitung statistik deskriptif, menjalankan PLS-SEM (skema pembobotan *path*, Mode A, bootstrapping 5.000 sub-sampel dengan CI BCa), dan menyimpan hasilnya ke Excel.

1. Buka [Google Colab](https://colab.research.google.com), buat notebook baru.
2. Salin isi `kode/Analisis_PLS-SEM_Shopee.py` per blok `# %%` ke sel-sel terpisah, atau unggah file tersebut.
3. Jalankan semua sel. Saat diminta, unggah file CSV data mentah.

Data mentah tidak disertakan karena memuat profil responden. Data tersebut tersedia berdasarkan permintaan kepada penulis korespondensi. Tanpa data mentah, hasil utama tetap dapat direplikasi dengan notebook seminr di bawah.

### 2. Verifikasi dengan seminr (R)

1. Unggah `kode/Verifikasi_seminr_Shopee.ipynb` ke Google Colab.
2. Pilih **Runtime → Change runtime type → R**.
3. Unggah `data/smartpls_A_n190.csv` ke panel Files.
4. Jalankan sel dari atas ke bawah. Pemasangan seminr dan bootstrapping masing-masing memerlukan beberapa menit.

## Hasil utama

| Hipotesis | β | t (Python) | t (seminr) | Keputusan |
|---|---|---|---|---|
| H1 EXP → CONF | 0,468 | 7,18 | 7,30 | Didukung |
| H2 EXP → SAT | 0,199 | 2,20 | 2,15 | Didukung |
| H3 CONF → SAT | 0,659 | 6,76 | 6,51 | Didukung |
| H4 CONF → EW | −0,034 | −0,43 | −0,43 | Tidak didukung |
| H5 SAT → EW | 0,716 | 9,47 | 9,68 | Didukung |
| H6 EW → REP | 0,697 | 17,67 | 17,36 | Didukung |

Koefisien jalur dan seluruh outer loading dari Python dan seminr identik sampai tiga desimal. Nilai t berbeda tipis karena sub-sampel bootstrap diambil secara acak dan kedua alat memakai metode interval yang berbeda (BCa di Python, persentil di seminr). Seed bootstrap: `20261102`.

## Lisensi

Kode dirilis dengan lisensi MIT. Data dalam folder `data/` dapat digunakan ulang dengan menyebutkan artikel di atas sebagai sumber.

## Kontak

[[Nama penulis korespondensi]], [[email]]
