import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.cluster import KMeans

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# Variabel global untuk menyimpan dataset hasil upload
df = None


# Fungsi Konversi Waktu
def convert_to_minutes(waktu):
    """
    Mengonversi waktu dalam format 'HH:MM' menjadi total menit.

    Contoh:
    08:30 -> 510

    Digunakan agar data waktu dapat diproses secara numerik
    dalam algoritma machine learning.
    """
    if isinstance(waktu, str):
        jam, menit = map(int, waktu.split(":"))
        return jam * 60 + menit
    return np.nan


def minutes_to_time(menit):
    """
    Mengonversi nilai menit kembali ke format 'HH:MM'.

    Fungsi ini hanya digunakan untuk keperluan tampilan data,
    bukan untuk proses pelatihan model.
    """
    if pd.isna(menit):
        return ""
    jam = int(menit // 60)
    menit = int(menit % 60)
    return f"{jam:02d}:{menit:02d}"


def hours_fraction_to_time(value):
    """
    Mengonversi nilai jam dalam bentuk desimal ke format 'HH:MM'.

    Contoh:
    2.5 -> 02:30

    Digunakan untuk menampilkan durasi penggunaan aplikasi
    secara lebih mudah dipahami oleh pengguna.
    """
    if pd.isna(value):
        return ""
    jam = int(value)
    menit = int((value - jam) * 60)
    return f"{jam:02d}:{menit:02d}"


# HALAMAN UTAMA
@app.route("/")
def index():
    """
    Menampilkan halaman utama aplikasi.
    """
    return render_template("index.html")


# UPLOAD DATA
@app.post("/upload")
def upload():
    """
    Menerima file CSV dari pengguna, melakukan preprocessing,
    serta menampilkan data dalam bentuk tabel.
    """
    global df

    # Membaca file CSV
    file = request.files["file"]
    df = pd.read_csv(file).dropna()

    # Reset index agar rapi saat ditampilkan
    df.index = range(1, len(df) + 1)

    # Pembulatan nilai numerik
    df["total_tugas_selesai"] = df["total_tugas_selesai"].round()
    df["rata_durasi_tugas"] = df["rata_durasi_tugas"].round()
    df["penggunaan_aplikasi_produktif"] = (
        df["penggunaan_aplikasi_produktif"].round(2)
    )

    # Konversi waktu masuk ke satuan menit
    df["waktu_masuk"] = df["waktu_masuk"].apply(convert_to_minutes)

    # Salinan data untuk ditampilkan
    df_display = df.copy()
    df_display["waktu_masuk"] = (
        df_display["waktu_masuk"].apply(minutes_to_time)
    )
    df_display["penggunaan_aplikasi_produktif"] = (
        df_display["penggunaan_aplikasi_produktif"]
        .apply(hours_fraction_to_time)
    )

    return render_template(
        "index.html",
        table=df_display.to_html(
            classes="table table-bordered",
            index=True
        )
    )

# PELATIHAN MODEL KLASIFIKASI
@app.post("/latih")
def latih():
    """
    Melatih model Decision Tree dan Random Forest
    untuk klasifikasi produktivitas karyawan,
    kemudian menampilkan hasil evaluasi dan
    feature importance.
    """
    global df

    if df is None:
        return "Data belum di-upload"

    # Fitur yang digunakan dalam pemodelan
    fitur = [
        "total_tugas_selesai",
        "rata_durasi_tugas",
        "penggunaan_aplikasi_produktif",
        "waktu_masuk"
    ]

    X = df[fitur]
    y = df["label_produktivitas"]

    # Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pembagian data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Model Decision Tree (Baseline)
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )

    # Model Random Forest (Ensemble)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )

    # Pelatihan model
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Prediksi data uji
    dt_pred = dt.predict(X_test)
    rf_pred = rf.predict(X_test)

    # Evaluasi performa model
    results = {
        "Decision Tree": {
            "accuracy": round(
                accuracy_score(y_test, dt_pred), 3
            ),
            "precision": round(
                precision_score(
                    y_test, dt_pred, average="macro"
                ), 3
            ),
            "recall": round(
                recall_score(
                    y_test, dt_pred, average="macro"
                ), 3
            ),
            "f1": round(
                f1_score(
                    y_test, dt_pred, average="macro"
                ), 3
            ),
        },
        "Random Forest": {
            "accuracy": round(
                accuracy_score(y_test, rf_pred), 3
            ),
            "precision": round(
                precision_score(
                    y_test, rf_pred, average="macro"
                ), 3
            ),
            "recall": round(
                recall_score(
                    y_test, rf_pred, average="macro"
                ), 3
            ),
            "f1": round(
                f1_score(
                    y_test, rf_pred, average="macro"
                ), 3
            ),
        }
    }

    # Feature importance dari Random Forest
    feat_importance = pd.DataFrame({
        "Fitur": fitur,
        "Kepentingan": rf.feature_importances_
    }).sort_values(
        by="Kepentingan",
        ascending=False
    )

    # Data untuk ditampilkan
    df_display = df.copy()
    df_display["waktu_masuk"] = (
        df_display["waktu_masuk"].apply(minutes_to_time)
    )
    df_display["penggunaan_aplikasi_produktif"] = (
        df_display["penggunaan_aplikasi_produktif"]
        .apply(hours_fraction_to_time)
    )

    return render_template(
        "index.html",
        table=df_display.to_html(
            classes="table table-bordered",
            index=True
        ),
        results=results,
        importance=feat_importance.to_html(
            classes="table table-bordered",
            index=False
        )
    )


# CLUSTERING PRODUKTIVITAS
@app.post("/cluster")
def cluster():
    """
    Melakukan clustering produktivitas karyawan
    menggunakan algoritma K-Means dan memberikan
    label produktivitas otomatis.
    """
    global df

    if df is None:
        return "Data belum di-upload"

    fitur_cluster = [
        "total_tugas_selesai",
        "rata_durasi_tugas",
        "penggunaan_aplikasi_produktif",
        "waktu_masuk"
    ]

    # Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(
        df[fitur_cluster]
    )

    # Model K-Means
    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    # Menentukan cluster
    df["Cluster"] = model.fit_predict(X_scaled)

    # Penilaian cluster berdasarkan bobot logis
    cluster_scores = (
        df.groupby("Cluster")
        .apply(lambda x: (
            x["total_tugas_selesai"].mean() * 0.4 +
            x["penggunaan_aplikasi_produktif"].mean() * 0.3 -
            x["rata_durasi_tugas"].mean() * 0.2 -
            x["waktu_masuk"].mean() * 0.1
        ))
    )

    # Pengurutan cluster
    ranking = (
        cluster_scores
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # Mapping label produktivitas
    mapping = {
        ranking[0]: "produktif",
        ranking[1]: "normal",
        ranking[2]: "tidak_produktif"
    }

    df["label_otomatis"] = df["Cluster"].map(mapping)

    # Data untuk ditampilkan
    df_display = df.copy()
    df_display["waktu_masuk"] = (
        df_display["waktu_masuk"].apply(minutes_to_time)
    )
    df_display["penggunaan_aplikasi_produktif"] = (
        df_display["penggunaan_aplikasi_produktif"]
        .apply(hours_fraction_to_time)
    )

    return render_template(
        "index.html",
        table=df_display.to_html(
            classes="table table-bordered",
            index=True
        )
    )

# MENJALANKAN APLIKASI
if __name__ == "__main__":
    app.run(debug=True)
