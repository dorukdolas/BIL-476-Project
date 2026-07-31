# BIL 476 - Data Mining Projesi

Bu repo, Bank Marketing veri seti üzerinde bir müşterinin vadeli mevduat açıp
açmayacağını tahmin etmek için yaptığım sınıflandırma çalışmasının kodlarını
içeriyor. Dört farklı algoritmayı (Decision Tree, Naive Bayes, k-NN, Random
Forest) karşılaştırdım ve `duration` özniteliğinin sonuçlara etkisini inceledim.

## Veri seti

- `data/bank.csv` (Kaggle - Bank Marketing, dengeli sürüm)
- 11162 satır, 16 öznitelik + hedef (`deposit`)
- Sınıflar dengeli (yaklaşık %47 yes)

## Klasörler

- `data/` - veri seti
- `src/` - kodlar (config, eda, experiments)
- `figures/` - üretilen grafikler
- `results/` - metrik tabloları (csv)
- `report/` - LaTeX raporu

## Nasıl çalıştırılır

Python 3.10 kullandım (3.14'te bazı kütüphaneler sorun çıkardı, o yüzden 3.10).
Önce gerekli kütüphaneleri kurun:

```bash
py -3.10 -m pip install -r requirements.txt
```

Sonra sırayla çalıştırın:

```bash
py -3.10 src/eda.py
py -3.10 src/experiments.py
```

`eda.py` veriyle ilgili grafik ve istatistikleri çıkarıyor. `experiments.py` ise
modelleri eğitip test ediyor ve rapordaki tablo/figürlerin hepsini üretiyor.
random_state=42 sabit olduğu için sonuçlar her çalıştırmada aynı çıkıyor.

## Sonuçlar

Random Forest iki senaryoda da en iyi sonucu verdi:

| Senaryo | Accuracy | ROC-AUC |
|---|---|---|
| Tüm öznitelikler | 0.858 | 0.921 |
| duration çıkarılmış | 0.729 | 0.786 |

`duration` görüşme süresi olduğu için aslında görüşme bitmeden bilinemiyor,
yani modele koyunca sonucu şişiriyor (bir çeşit leakage). Çıkarınca AUC 0.921'den
0.786'ya düşüyor. Raporda bu kısmı ayrıca tartıştım.
