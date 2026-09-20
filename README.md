# machine_learning
# Machine Learning Çalışmaları

Python ve Scikit-learn kullanarak yaptığım makine öğrenmesi çalışmalarını içerir.

## Audi A1 Fiyat Tahmini

Audi A1 araç verileri kullanılarak çoklu doğrusal regresyon modeli oluşturuldu.

### Kullanılan özellikler

* Üretim yılı
* Kilometre
* Motor hacmi
* Motor gücü
* Kasa tipi
* Vites türü
* Yakıt türü
* Sahip sayısı

### Uygulanan işlemler

* CSV verisini Pandas ile okuma
* Gereksiz sütunları kaldırma
* Sütun isimlerini düzenleme
* Motor hacmini sayısal veriye dönüştürme
* Kategorik verileri `get_dummies()` ile kodlama
* Veriyi eğitim ve test kümelerine ayırma
* `LinearRegression` ile model eğitme
* Model başarısını \(R^2\) skoru ile ölçme
* Yeni araç için fiyat tahmini yapma

## Kullanılan kütüphaneler

```text
pandas
scikit-learn
matplotlib
numpy
```

## Çalıştırma

```bash
pip install -r requirements.txt
python arac_fiyat_tahmini.py
```

Bu repo, makine öğrenmesi öğrenme sürecinde geliştirilen yeni uygulamalarla güncellenecektir.
