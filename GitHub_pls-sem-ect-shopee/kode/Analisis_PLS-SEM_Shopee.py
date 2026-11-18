# %% [markdown]
# # Analisis PLS-SEM Skripsi: Pengguna Shopee (EXP, CONF, SAT, eWOM, REP)
# 
# Notebook ini mengolah data **mentah Google Form** sampai menghasilkan tabel hasil PLS-SEM (dataset A, n = 190).
# 
# **Cara pakai:** jalankan sel dari atas ke bawah (Runtime → Run all). Saat sel *Unggah data* berjalan, pilih file CSV hasil ekspor Google Form Anda (`... - Form Responses 1.csv`).
# 
# **Catatan penting:** notebook ini adalah implementasi PLS-SEM di Python (algoritma Lohmöller dengan skema pembobotan *path*, Mode A), dan menjadi satu-satunya alat analisis (tanpa SmartPLS). Seluruh keluaran yang lazim dilaporkan di skripsi tersedia: loading, reliabilitas, validitas diskriminan, VIF, jalur struktural dengan bootstrapping, efek tidak langsung, R², f², SRMR, Q² (blindfolding), dan PLSpredict. Hasil estimasi sudah diverifikasi dengan paket R `seminr` (lihat `Verifikasi_seminr_Shopee.ipynb`): koefisien jalur dan outer loading identik sampai tiga desimal (lihat catatan di akhir).

# %% [markdown]
# ## Langkah 1. Pengaturan awal dan definisi model
# Sel ini memuat pustaka dan mendefinisikan model: 5 konstruk reflektif (masing-masing 3 indikator) dan 6 jalur hipotesis (H1 sampai H6).

# %%
import numpy as np
import pandas as pd
import time
from scipy import stats
from IPython.display import display

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)

SEED = 20261102      # seed tetap agar hasil bootstrap dapat diulang
N_BOOT = 5000        # jumlah sub-sampel bootstrap (sama dengan SmartPLS)

# --- Definisi model ---
LV = ['EXP', 'CONF', 'SAT', 'EW', 'REP']
IND = {lv: [f'{lv}{i}' for i in (1, 2, 3)] for lv in LV}
ITEMS = [c for lv in LV for c in IND[lv]]
PATHS = [('EXP', 'CONF'),   # H1
         ('EXP', 'SAT'),    # H2
         ('CONF', 'SAT'),   # H3
         ('CONF', 'EW'),    # H4
         ('SAT', 'EW'),     # H5
         ('EW', 'REP')]     # H6
HIP = {f'{a} → {b}': f'H{i+1}' for i, (a, b) in enumerate(PATHS)}
BLK = {lv: [ITEMS.index(c) for c in IND[lv]] for lv in LV}
PRED = {lv: [LV.index(a) for a, b in PATHS if b == lv] for lv in LV}
SUCC = {lv: [LV.index(b) for a, b in PATHS if a == lv] for lv in LV}
ENDO = [lv for lv in LV if PRED[lv]]
print('Model siap:', len(LV), 'konstruk,', len(ITEMS), 'indikator,', len(PATHS), 'jalur')

# %% [markdown]
# ## Langkah 2. Unggah data mentah Google Form
# Di Google Colab akan muncul tombol *Choose Files*. Pilih file CSV ekspor Google Form (197 baris).

# %%
try:
    from google.colab import files
    unggahan = files.upload()
    NAMA_FILE = list(unggahan.keys())[0]
except ImportError:
    # jika dijalankan di luar Colab: letakkan file di folder yang sama dan isi namanya di sini
    NAMA_FILE = 'Kuesioner_Penelitian_Pengalaman_Pengguna_Shopee___Jawaban__-_Form_Responses_1.csv'
print('File dipakai:', NAMA_FILE)

# %% [markdown]
# ## Langkah 3. Membaca data dan memberi kode kolom
# Kolom Google Form yang panjang diganti kode pendek berdasarkan urutan: 1 timestamp, 4 pertanyaan profil, lalu 15 butir (EXP1–3, CONF1–3, SAT1–3, EW1–3, REP1–3). Setiap responden diberi `ID` sesuai urutan baris asli.

# %%
raw = pd.read_csv(NAMA_FILE)
raw.columns = [c.strip() for c in raw.columns]
print('Bentuk data mentah:', raw.shape, '(harus 197 baris x 20 kolom)')
assert raw.shape[1] == 20, 'Jumlah kolom tidak 20: periksa urutan kolom kuesioner.'

df = raw.copy()
df.columns = ['Timestamp', 'Usia', 'Jenis_Kelamin', 'Frekuensi_3bln', 'Baca_Ulasan'] + ITEMS
df.insert(0, 'ID', range(1, len(df) + 1))
display(df.head())

# %% [markdown]
# ## Langkah 4. Audit kualitas data
# Memeriksa nilai kosong, rentang jawaban, periode pengisian, jawaban ganda, dan pola jawaban seragam (*straight-lining*).

# %%
print('Total nilai kosong:', int(df.isna().sum().sum()))
print('Rentang jawaban butir (harus 1-5):', int(df[ITEMS].min().min()), '-', int(df[ITEMS].max().max()))

ts_txt = df['Timestamp'].astype(str).str.replace(' GMT+7', '', regex=False)
tahun_dulu = ts_txt.str.contains('^[0-9]{4}/', regex=True).all()     # True jika format 2026/09/30 ...
fmt = '%Y/%m/%d %I:%M:%S %p' if tahun_dulu else '%d/%m/%Y %H:%M:%S'
ts = pd.to_datetime(ts_txt, format=fmt, errors='coerce')
print('Periode pengisian:', ts.min(), 's.d.', ts.max())
print('Respons per hari (5 teratas):')
print(ts.dt.date.value_counts().head(5))

df['Straight_lining'] = df[ITEMS].std(axis=1) == 0
print('\nBaris dengan seluruh jawaban sama persis (selain timestamp):',
      int(df.duplicated(subset=[c for c in df.columns if c not in ('ID', 'Timestamp', 'Straight_lining')]).sum()))
print('Responden straight-lining (15 butir seragam):', int(df['Straight_lining'].sum()))
print('\nFrekuensi berbelanja di Shopee (3 bulan terakhir):')
print(df['Frekuensi_3bln'].value_counts())

# %% [markdown]
# ## Langkah 5. Membentuk dataset A
# Dataset A = semua responden yang **pernah** berbelanja di Shopee dalam 3 bulan terakhir (jawaban "Tidak pernah" dikeluarkan karena tidak memenuhi kriteria). Responden straight-lining **tidak** dihapus pada dataset A. File CSV yang dihasilkan dapat diimpor ke SmartPLS 4.

# %%
A = df[df['Frekuensi_3bln'] != 'Tidak pernah'].copy()
print('Dataset A:', len(A), 'responden | straight-lining di dalamnya:', int(A['Straight_lining'].sum()))

A[['ID'] + ITEMS].to_csv('smartpls_A_n190.csv', index=False)
A.to_csv('data_bersih_A_lengkap.csv', index=False)
print('Tersimpan: smartpls_A_n190.csv (untuk SmartPLS) dan data_bersih_A_lengkap.csv')

# %% [markdown]
# ## Langkah 6. Statistik deskriptif dan uji normalitas
# Profil responden, rata-rata dan simpangan baku tiap butir, skewness, kurtosis, serta uji Mardia. Data yang tidak normal multivariat membenarkan pemakaian PLS-SEM.

# %%
for kolom in ['Usia', 'Jenis_Kelamin', 'Frekuensi_3bln', 'Baca_Ulasan']:
    t = A[kolom].value_counts().rename('Jumlah').to_frame()
    t['Persen'] = (t['Jumlah'] / len(A) * 100).round(1)
    print('\n', kolom); display(t)

desk = pd.DataFrame({
    'Mean': A[ITEMS].mean(), 'SD': A[ITEMS].std(),
    'Min': A[ITEMS].min(), 'Max': A[ITEMS].max(),
    'Skewness': A[ITEMS].apply(stats.skew), 'Kurtosis': A[ITEMS].apply(stats.kurtosis)}).round(3)
display(desk)

X = A[ITEMS].values.astype(float)
n, p = X.shape
Xc = X - X.mean(0)
S = np.cov(X, rowvar=False, bias=True)
D = Xc @ np.linalg.inv(S) @ Xc.T
b1 = (D ** 3).sum() / n ** 2
b2 = (np.diag(D) ** 2).mean()
chi = n * b1 / 6
df_chi = p * (p + 1) * (p + 2) / 6
zk = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
print(f'Mardia skewness: chi2({df_chi:.0f}) = {chi:.1f}, p = {1 - stats.chi2.cdf(chi, df_chi):.4f}')
print(f'Mardia kurtosis: b2 = {b2:.1f}, z = {zk:.1f}, p = {2 * (1 - stats.norm.cdf(abs(zk))):.4f}')

# %% [markdown]
# ## Langkah 7. Mesin PLS-SEM
# Sel ini berisi algoritma PLS-SEM (skema pembobotan *path*, Mode A untuk konstruk reflektif, toleransi 1e-7, maksimum 3000 iterasi) beserta fungsi reliabilitas, validitas, statistik struktural, dan *bootstrapping* (BCa). Cukup dijalankan, tidak perlu diubah.

# %%
def _std(a):
    return (a - a.mean(0)) / a.std(0, ddof=1)


def fit(X, tol=1e-7, maxit=3000):
    n = X.shape[0]
    Xs = _std(X)
    W = [np.ones(3) for _ in LV]

    def scores(W):
        Y = np.column_stack([Xs[:, BLK[lv]] @ W[j] for j, lv in enumerate(LV)])
        return Y, Y.std(0, ddof=1)

    Y, sd = scores(W)
    W = [W[j] / sd[j] for j in range(5)]
    Y = _std(Y)
    for it in range(maxit):
        Z = np.zeros_like(Y)
        for j, lv in enumerate(LV):
            z = np.zeros(n)
            if PRED[lv]:
                Pm = Y[:, PRED[lv]]
                beta = np.linalg.lstsq(Pm, Y[:, j], rcond=None)[0]
                z += Pm @ beta
            for s in SUCC[lv]:
                z += np.corrcoef(Y[:, j], Y[:, s])[0, 1] * Y[:, s]
            Z[:, j] = (z - z.mean()) / z.std(ddof=1)
        Wn = [(Xs[:, BLK[lv]] * Z[:, [j]]).sum(0) / (n - 1) for j, lv in enumerate(LV)]
        Yn, sd = scores(Wn)
        Wn = [Wn[j] / sd[j] for j in range(5)]
        delta = max(np.abs(Wn[j] - W[j]).max() for j in range(5))
        W, Y = Wn, _std(Yn)
        if delta < tol:
            break
    load = [np.array([np.corrcoef(Xs[:, c], Y[:, j])[0, 1] for c in BLK[lv]]) for j, lv in enumerate(LV)]
    B = np.zeros((5, 5))
    r2 = {}
    for j, lv in enumerate(LV):
        if PRED[lv]:
            Pm = Y[:, PRED[lv]]
            beta = np.linalg.lstsq(Pm, Y[:, j], rcond=None)[0]
            for k, pidx in enumerate(PRED[lv]):
                B[pidx, j] = beta[k]
            resid = Y[:, j] - Pm @ beta
            r2[lv] = 1 - (resid ** 2).sum() / (n - 1)
    return dict(W=W, load=load, Y=Y, B=B, r2=r2, iters=it + 1, Xs=Xs)


def all_paths(src, dst, path=None):
    path = path or [src]
    out = []
    for nxt in SUCC[LV[src]]:
        if nxt in path:
            continue
        pth = path + [nxt]
        if nxt == dst:
            out.append(pth)
        else:
            out += all_paths(nxt, dst, pth)
    return out


def indirect_specs(B):
    specs = {}
    for s in range(5):
        for d in range(5):
            if s == d:
                continue
            for pth in all_paths(s, d):
                if len(pth) >= 3:
                    specs[' → '.join(LV[i] for i in pth)] = np.prod([B[pth[k], pth[k + 1]] for k in range(len(pth) - 1)])
    return specs


def total_effects(B):
    T = {}
    for s in range(5):
        for d in range(5):
            if s != d:
                ps = all_paths(s, d)
                if ps:
                    T[(LV[s], LV[d])] = sum(np.prod([B[q[k], q[k + 1]] for k in range(len(q) - 1)]) for q in ps)
    return T


def htmt(R):
    out = np.zeros((5, 5))
    mono = [R[np.ix_(BLK[lv], BLK[lv])][np.triu_indices(3, 1)].mean() for lv in LV]
    for a in range(5):
        for b in range(5):
            if a != b:
                out[a, b] = R[np.ix_(BLK[LV[a]], BLK[LV[b]])].mean() / np.sqrt(mono[a] * mono[b])
    return out


def reliability(res, R):
    rows = []
    for j, lv in enumerate(LV):
        l = res['load'][j]
        k = len(l)
        sub = R[np.ix_(BLK[lv], BLK[lv])]
        rbar = sub[np.triu_indices(k, 1)].mean()
        alpha = k * rbar / (1 + (k - 1) * rbar)
        cr = l.sum() ** 2 / (l.sum() ** 2 + (1 - l ** 2).sum())
        w = res['W'][j]
        ww = np.outer(w, w)
        rho_a = (w @ w) ** 2 * (w @ (sub - np.diag(np.diag(sub))) @ w) / (w @ (ww - np.diag(np.diag(ww))) @ w)
        rows.append(dict(Konstruk=lv, Cronbach_alpha=alpha, rho_A=rho_a, rho_C_CR=cr, AVE=(l ** 2).mean()))
    return pd.DataFrame(rows).set_index('Konstruk')


def structural_stats(res):
    Y, n = res['Y'], res['Y'].shape[0]
    rows_r, rows_f = [], []
    for lv in ENDO:
        j = LV.index(lv)
        r2 = res['r2'][lv]
        k = len(PRED[lv])
        rows_r.append(dict(Konstruk=lv, R2=r2, R2_adj=1 - (1 - r2) * (n - 1) / (n - k - 1)))
        Pj = PRED[lv]
        for pidx in Pj:
            others = [q for q in Pj if q != pidx]
            if others:
                Po = Y[:, others]
                bt = np.linalg.lstsq(Po, Y[:, j], rcond=None)[0]
                r2x = 1 - ((Y[:, j] - Po @ bt) ** 2).sum() / (n - 1)
                bb = np.linalg.lstsq(Po, Y[:, pidx], rcond=None)[0]
                r2p = 1 - ((Y[:, pidx] - Po @ bb) ** 2).sum() / (n - 1)
                vif = 1 / (1 - r2p)
            else:
                r2x, vif = 0.0, 1.0
            rows_f.append(dict(Jalur=f'{LV[pidx]} → {lv}', f2=(res['r2'][lv] - r2x) / (1 - res['r2'][lv]), VIF=vif))
    return pd.DataFrame(rows_r).set_index('Konstruk'), pd.DataFrame(rows_f).set_index('Jalur')


def stat_vector(X):
    res = fit(X)
    B = res['B']
    R = np.corrcoef(X, rowvar=False)
    H = htmt(R)
    return dict(
        paths=np.array([B[LV.index(a), LV.index(b)] for a, b in PATHS]),
        spec=indirect_specs(B), tot=total_effects(B),
        htmt=np.array([H[a, b] for a in range(5) for b in range(a + 1, 5)]),
        load=np.concatenate(res['load']))


def bca_ci(boot, theta, jack, alpha=0.05):
    boot = np.asarray(boot)
    z0 = stats.norm.ppf(np.clip((boot < theta).mean() + 0.5 * (boot == theta).mean(), 1e-6, 1 - 1e-6))
    jm = jack.mean()
    den = 6 * (((jm - jack) ** 2).sum()) ** 1.5
    a = ((jm - jack) ** 3).sum() / den if den > 0 else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = stats.norm.ppf(q)
        out.append(np.percentile(boot, 100 * stats.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))))
    return tuple(out)


def bootstrap(X, B=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    base = stat_vector(X)
    reps = []
    for _ in range(B):
        xs = X[rng.integers(0, n, n)]
        if (xs.std(0, ddof=1) == 0).any():
            continue
        try:
            reps.append(stat_vector(xs))
        except Exception:
            continue
    jk = [stat_vector(np.delete(X, i, axis=0)) for i in range(n)]
    return base, reps, jk


def pval(t, n):
    return 2 * (1 - stats.t.cdf(abs(t), n - 1))

def srmr_calc(X, res):
    R = np.corrcoef(X, rowvar=False)
    lam = np.concatenate(res['load'])
    B = res['B']
    def implied(Phi):
        S = np.zeros((15, 15))
        for a in range(5):
            for b in range(5):
                for i in BLK[LV[a]]:
                    for k in BLK[LV[b]]:
                        S[i, k] = 1.0 if i == k else lam[i] * lam[k] * (1.0 if a == b else Phi[a, b])
        return S
    Phi_s = np.corrcoef(res['Y'], rowvar=False)           # model jenuh
    Phi_e = np.eye(5)                                      # model terestimasi (implikasi jalur)
    for j in range(5):
        pred = PRED[LV[j]]
        if pred:
            beta = B[pred, j]
            for m in range(j):
                v = sum(beta[k] * Phi_e[pred[k], m] for k in range(len(pred)))
                Phi_e[j, m] = Phi_e[m, j] = v
    iu = np.triu_indices(15)
    out = {}
    for nama, Phi in (('Model jenuh', Phi_s), ('Model terestimasi', Phi_e)):
        d = (R - implied(Phi))[iu]
        out[nama] = float(np.sqrt((d ** 2).mean()))
    return out

def blindfolding(X, D=7):
    n = X.shape[0]
    rows = []
    for lv in ENDO:
        j = LV.index(lv); cols = BLK[lv]; K = len(cols)
        sse = np.zeros(K); sso = np.zeros(K)
        pos = np.arange(n)[:, None] * K + np.arange(K)[None, :]
        for g in range(D):
            omit = (pos % D) == g
            Xm = X.copy()
            for k, c in enumerate(cols):
                Xm[omit[:, k], c] = X[~omit[:, k], c].mean()
            r = fit(Xm)
            L_hat = r['Y'][:, PRED[lv]] @ r['B'][PRED[lv], j]
            lam = r['load'][j]
            for k, c in enumerate(cols):
                keep = ~omit[:, k]
                mk = X[keep, c].mean(); sk = Xm[:, c].std(ddof=1)
                xhat = mk + sk * lam[k] * L_hat
                sse[k] += ((X[omit[:, k], c] - xhat[omit[:, k]]) ** 2).sum()
                sso[k] += ((X[omit[:, k], c] - mk) ** 2).sum()
        rows.append(dict(Konstruk=lv, SSO=sso.sum(), SSE=sse.sum(), Q2=1 - sse.sum() / sso.sum()))
    return pd.DataFrame(rows)

def plspredict(X, folds=10, reps=10, seed=SEED):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    tgt = [(lv, c) for lv in ENDO for c in BLK[lv]]
    T = len(tgt)
    res_rep = []
    for rep in range(reps):
        perm = rng.permutation(n)
        P_pls = np.zeros((n, T)); P_lm = np.zeros((n, T)); P_nv = np.zeros((n, T))
        for idx in np.array_split(perm, folds):
            tr = np.setdiff1d(np.arange(n), idx)
            Xtr, Xh = X[tr], X[idx]
            r = fit(Xtr)
            m, s = Xtr.mean(0), Xtr.std(0, ddof=1)
            Zh = (Xh - m) / s
            S = np.column_stack([Zh[:, BLK[lv]] @ r['W'][j] for j, lv in enumerate(LV)])
            for t, (lv, c) in enumerate(tgt):
                j = LV.index(lv); k = BLK[lv].index(c)
                lhat = S[:, PRED[lv]] @ r['B'][PRED[lv], j]
                P_pls[idx, t] = m[c] + s[c] * r['load'][j][k] * lhat
                P_nv[idx, t] = m[c]
                ant = [i for p in PRED[lv] for i in BLK[LV[p]]]
                A = np.column_stack([np.ones(len(tr)), Xtr[:, ant]])
                beta = np.linalg.lstsq(A, Xtr[:, c], rcond=None)[0]
                P_lm[idx, t] = np.column_stack([np.ones(len(idx)), Xh[:, ant]]) @ beta
        Y = X[:, [c for _, c in tgt]]
        res_rep.append(dict(
            rmse_pls=np.sqrt(((Y - P_pls) ** 2).mean(0)), rmse_lm=np.sqrt(((Y - P_lm) ** 2).mean(0)),
            mae_pls=np.abs(Y - P_pls).mean(0), mae_lm=np.abs(Y - P_lm).mean(0),
            q2=1 - ((Y - P_pls) ** 2).sum(0) / ((Y - P_nv) ** 2).sum(0)))
    avg = {k: np.mean([r[k] for r in res_rep], axis=0) for k in res_rep[0]}
    tab = pd.DataFrame({'Konstruk': [lv for lv, _ in tgt], 'Indikator': [ITEMS[c] for _, c in tgt],
                        'Q2_predict': avg['q2'], 'RMSE_PLS': avg['rmse_pls'], 'RMSE_LM': avg['rmse_lm'],
                        'MAE_PLS': avg['mae_pls'], 'MAE_LM': avg['mae_lm']})
    tab['PLS_lebih_baik_dari_LM'] = tab['RMSE_PLS'] < tab['RMSE_LM']
    return tab


def atur_model(paths):
    """Mengubah daftar jalur model (dipakai untuk analisis tambahan). Urutan LV harus tetap topologis."""
    PATHS[:] = paths
    for lv in LV:
        PRED[lv] = [LV.index(a) for a, b in paths if b == lv]
        SUCC[lv] = [LV.index(b) for a, b in paths if a == lv]
    ENDO[:] = [lv for lv in LV if PRED[lv]]

print('Mesin PLS-SEM siap.')

# %% [markdown]
# ## Langkah 8. Estimasi model (tanpa bootstrap)
# Memastikan algoritma konvergen, lalu melihat *loading*, koefisien jalur, dan R².

# %%
X = A[ITEMS].values.astype(float)
res = fit(X)
print('Konvergen dalam', res['iters'], 'iterasi')
display(pd.DataFrame({'Indikator': ITEMS, 'Loading': np.concatenate(res['load'])}).round(3).set_index('Indikator').T)
display(pd.DataFrame(res['B'], index=LV, columns=LV).round(3))
print('R2:', {k: round(v, 3) for k, v in res['r2'].items()})

# %% [markdown]
# ## Langkah 9. Bootstrapping (5.000 sub-sampel) dan seluruh tabel hasil
# Langkah ini memerlukan sekitar 30 sampai 60 detik. Hasilnya: model pengukuran, validitas diskriminan, model struktural, efek tidak langsung, dan keputusan hipotesis.

# %%
def analisis(d):
    X = d[ITEMS].values.astype(float)
    n = len(X)
    res = fit(X)
    R = np.corrcoef(X, rowvar=False)
    t0 = time.time()
    base, reps, jk = bootstrap(X)
    print(f'Bootstrap selesai: {len(reps)} replikasi valid, {time.time() - t0:.0f} detik')
    out = {'n': n}

    # --- Model pengukuran
    L = np.concatenate(res['load'])
    bl = np.array([r['load'] for r in reps])
    se = bl.std(0, ddof=1)
    out['Loading'] = pd.DataFrame({'Konstruk': [lv for lv in LV for _ in range(3)], 'Indikator': ITEMS,
                                   'Loading': L, 'SE': se, 't': L / se, 'p': [pval(l / s, n) for l, s in zip(L, se)]})
    rel = reliability(res, R)
    out['Reliabilitas'] = rel.reset_index()

    # --- Validitas diskriminan
    cm = np.corrcoef(res['Y'], rowvar=False)
    fl = pd.DataFrame(cm.copy(), index=LV, columns=LV)   # .copy(): cegah diagonal cm ikut berubah (pandas versi lama berbagi memori)
    for lv in LV:
        fl.loc[lv, lv] = np.sqrt(rel.loc[lv, 'AVE'])
    out['Fornell_Larcker'] = fl.reset_index().rename(columns={'index': 'Konstruk'})
    H = htmt(R)
    hb = np.array([r['htmt'] for r in reps])
    pairs = [(LV[a], LV[b]) for a in range(5) for b in range(a + 1, 5)]
    out['HTMT'] = pd.DataFrame([dict(Pasangan=f'{a} – {b}', HTMT=H[LV.index(a), LV.index(b)],
                                     CI_2_5=np.percentile(hb[:, k], 2.5), CI_97_5=np.percentile(hb[:, k], 97.5))
                                for k, (a, b) in enumerate(pairs)])
    Xs, Y = res['Xs'], res['Y']
    out['Cross_loading'] = pd.DataFrame([[np.corrcoef(Xs[:, i], Y[:, j])[0, 1] for j in range(5)] for i in range(15)],
                                        index=ITEMS, columns=LV).reset_index().rename(columns={'index': 'Indikator'})
    out['VIF_kolinearitas_penuh'] = pd.DataFrame({'Konstruk': LV, 'Full_VIF': np.diag(np.linalg.inv(cm))})
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]
    out['Harman_pct'] = ev[0] / 15 * 100

    # --- Model struktural
    sp = np.array([r['paths'] for r in reps])
    jp = np.array([r['paths'] for r in jk])
    r2tab, ftab = structural_stats(res)
    rows = []
    for k, (a, b) in enumerate(PATHS):
        est = base['paths'][k]
        s = sp[:, k].std(ddof=1)
        t = est / s
        lo, hi = bca_ci(sp[:, k], est, jp[:, k])
        pv = pval(t, n)
        rows.append(dict(Hipotesis=HIP.get(f'{a} → {b}', 'Tambahan'), Jalur=f'{a} → {b}', Beta=est, SE=s, t=t, p=pv,
                         CI_2_5=lo, CI_97_5=hi, f2=ftab.loc[f'{a} → {b}', 'f2'], VIF=ftab.loc[f'{a} → {b}', 'VIF'],
                         Keputusan='Didukung' if pv < 0.05 else 'Tidak didukung'))
    out['Jalur_struktural'] = pd.DataFrame(rows)
    out['R2'] = r2tab.reset_index()

    # --- Efek tidak langsung dan total
    rows = []
    for key, est in base['spec'].items():
        bv = np.array([r['spec'][key] for r in reps])
        jv = np.array([r['spec'][key] for r in jk])
        s = bv.std(ddof=1)
        t = est / s
        lo, hi = bca_ci(bv, est, jv)
        rows.append(dict(Jalur=key, Efek=est, SE=s, t=t, p=pval(t, n), CI_2_5=lo, CI_97_5=hi))
    out['Efek_tidak_langsung'] = pd.DataFrame(rows)
    rows = []
    for (a, b), est in base['tot'].items():
        bv = np.array([r['tot'][(a, b)] for r in reps])
        jv = np.array([r['tot'][(a, b)] for r in jk])
        s = bv.std(ddof=1)
        t = est / s
        lo, hi = bca_ci(bv, est, jv)
        rows.append(dict(Jalur=f'{a} → {b}', Efek_total=est, SE=s, t=t, p=pval(t, n), CI_2_5=lo, CI_97_5=hi))
    out['Efek_total'] = pd.DataFrame(rows)

    # --- Kecocokan model, Q2 (blindfolding), PLSpredict
    sr = srmr_calc(X, res)
    out['SRMR'] = pd.DataFrame({'Model': list(sr.keys()), 'SRMR': list(sr.values())})
    out['Q2_blindfolding'] = blindfolding(X, D=7)
    pp = plspredict(X, folds=10, reps=10)
    out['PLSpredict'] = pp
    ringkas = pp.groupby('Konstruk', sort=False).agg(Q2_predict=('Q2_predict', 'mean'),
                                                      indikator_PLS_lebih_baik=('PLS_lebih_baik_dari_LM', 'sum'),
                                                      jumlah_indikator=('PLS_lebih_baik_dari_LM', 'size')).reset_index()
    def kategori(r):
        if r.Q2_predict <= 0 or r.indikator_PLS_lebih_baik == 0: return 'Tidak ada daya prediksi'
        if r.indikator_PLS_lebih_baik == r.jumlah_indikator: return 'Tinggi'
        if r.indikator_PLS_lebih_baik > r.jumlah_indikator / 2: return 'Sedang'
        return 'Rendah'
    ringkas['Daya_prediksi_out_of_sample'] = ringkas.apply(kategori, axis=1)
    out['PLSpredict_ringkasan'] = ringkas
    return out


HASIL_A = analisis(A)

# %% [markdown]
# ## Langkah 10. Membaca hasil
# Ambang yang dipakai (Hair et al., 2022; Henseler et al., 2015; Kock, 2015): *loading* ≥ 0,708; CR dan rho_A ≥ 0,70; AVE ≥ 0,50; HTMT < 0,90 dan CI tidak memuat 1; VIF < 3 (kolinearitas penuh < 3,3); R² 0,25/0,50/0,75 lemah/moderat/kuat.

# %%
H = HASIL_A
print('=== MODEL PENGUKURAN ===')
display(H['Loading'].round(3))
display(H['Reliabilitas'].round(3))
print('\n=== VALIDITAS DISKRIMINAN ===')
display(H['Fornell_Larcker'].round(3))
display(H['HTMT'].round(3))
display(H['VIF_kolinearitas_penuh'].round(2))
print(f"Harman single-factor: {H['Harman_pct']:.1f}% (ambang 50%)")

# %%
print('=== MODEL STRUKTURAL (keputusan hipotesis) ===')
display(H['Jalur_struktural'].round(3))
display(H['R2'].round(3))
print('\n=== EFEK TIDAK LANGSUNG ===')
display(H['Efek_tidak_langsung'].round(3))
print('\n=== EFEK TOTAL ===')
display(H['Efek_total'].round(3))
print('\n=== KECOCOKAN MODEL (SRMR) ===')
display(H['SRMR'].round(3))
print('SRMR < 0,08 baik; < 0,10 masih dapat diterima. SRMR model terestimasi dihitung dengan pendekatan model-implied common factor, jadi tafsirkan dengan hati-hati.')
print('\n=== Q2 (BLINDFOLDING, omission distance 7) ===')
display(H['Q2_blindfolding'].round(3))
print('Q2 > 0 = relevansi prediktif; 0,02 / 0,15 / 0,35 = kecil / sedang / besar.')
print('\n=== PLSpredict (10 fold x 10 ulangan) ===')
display(H['PLSpredict'].round(3))
display(H['PLSpredict_ringkasan'].round(3))
print('Tinggi = semua indikator RMSE PLS < LM; Sedang = mayoritas; Rendah = minoritas (Shmueli et al., 2019; Hair et al., 2022).')

# %% [markdown]
# ## Langkah 11. Analisis tambahan: model alternatif dengan jalur langsung ke REP
# Model hipotesis hanya memperbolehkan reputasi (REP) dijelaskan oleh eWOM. Jika kepuasan dan konfirmasi sebenarnya berhubungan langsung dengan reputasi, pengaruh eWOM dapat terlihat terlalu besar. Sel ini menambahkan jalur SAT → REP dan CONF → REP **hanya sebagai pemeriksaan ketahanan** (bukan hipotesis) dan membandingkan hasilnya. Model dikembalikan ke semula setelah selesai.

# %%
MODEL_ASLI = list(PATHS)
atur_model(MODEL_ASLI + [('SAT', 'REP'), ('CONF', 'REP')])
HASIL_ALT = analisis(A)
atur_model(MODEL_ASLI)          # kembalikan ke model hipotesis
print('Model dikembalikan ke:', PATHS)

print('=== Model alternatif (jalur tambahan SAT → REP dan CONF → REP) ===')
display(HASIL_ALT['Jalur_struktural'][['Hipotesis', 'Jalur', 'Beta', 'SE', 't', 'p', 'CI_2_5', 'CI_97_5']].round(3))
display(HASIL_ALT['R2'].round(3))
display(HASIL_ALT['SRMR'].round(3))

# %% [markdown]
# ## Langkah 12 (opsional). Analisis sensitivitas tanpa responden straight-lining
# Dataset B = dataset A tanpa responden yang menjawab seragam di semua 15 butir. Dipakai untuk melihat apakah kesimpulan berubah.

# %%
B_data = A[~A['Straight_lining']].copy()
print('Dataset B:', len(B_data), 'responden')
B_data[['ID'] + ITEMS].to_csv('smartpls_B_n155.csv', index=False)
HASIL_B = analisis(B_data)
perbandingan = H['Jalur_struktural'][['Hipotesis', 'Jalur', 'Beta', 'p', 'Keputusan']].merge(
    HASIL_B['Jalur_struktural'][['Hipotesis', 'Beta', 'p', 'Keputusan']], on='Hipotesis', suffixes=('_A', '_B'))
display(perbandingan.round(3))

# %% [markdown]
# ## Langkah 13. Ekspor hasil ke Excel
# Semua tabel disimpan dalam satu file Excel (satu sheet per tabel) lalu diunduh.

# %%
def simpan_excel(hasil, nama_file):
    with pd.ExcelWriter(nama_file, engine='openpyxl') as xw:
        for nama, tabel in hasil.items():
            if isinstance(tabel, pd.DataFrame):
                tabel.to_excel(xw, sheet_name=nama[:31], index=False)

simpan_excel(HASIL_A, 'hasil_PLS-SEM_dataset_A.xlsx')
for nama_var, nama_berkas in (('HASIL_B', 'hasil_PLS-SEM_dataset_B.xlsx'), ('HASIL_ALT', 'hasil_PLS-SEM_model_alternatif.xlsx')):
    if nama_var in globals():
        simpan_excel(globals()[nama_var], nama_berkas)

try:
    from google.colab import files
    for f in ['hasil_PLS-SEM_dataset_A.xlsx', 'smartpls_A_n190.csv', 'data_bersih_A_lengkap.csv']:
        files.download(f)
except ImportError:
    print('Di luar Colab: file tersimpan di folder kerja.')

# %% [markdown]
# ## Catatan
# - Cara menulis di Bab 3: "Analisis data dilakukan dengan PLS-SEM menggunakan implementasi algoritma Lohmöller (1989) dengan skema pembobotan path (Mode A untuk konstruk reflektif) pada Python di Google Colab; signifikansi diuji dengan bootstrapping 5.000 sub-sampel (BCa 95%)." Lampirkan kode notebook.
# - Validasi implementasi: (1) algoritma konvergen ke titik tetap (satu iterasi tambahan tidak mengubah bobot); (2) pemulihan struktur pada data simulasi berstruktur diketahui; (3) diverifikasi dengan paket R `seminr` (notebook `Verifikasi_seminr_Shopee.ipynb`): koefisien jalur dan ke-15 outer loading identik sampai tiga desimal, sedangkan nilai t bootstrap berbeda tipis karena sub-sampel acak dan metode interval yang berbeda (BCa di Python, persentil di seminr).
# - SRMR model terestimasi dihitung dengan pendekatan common-factor-implied; Q² memakai blindfolding (Tenenhaus et al., 2005) dengan omission distance 7; PLSpredict memakai pendekatan antecedent langsung dengan 10 fold x 10 ulangan.
# - f² dihitung dari R² model OLS pada skor laten (pendekatan), jadi dapat sedikit berbeda dari perangkat lain.
# - Rujukan: Hair et al. (2022); Henseler, Ringle, & Sarstedt (2015); Kock (2015); Dijkstra & Henseler (2015).

