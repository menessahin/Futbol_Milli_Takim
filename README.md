# Türkiye Milli Takım | Kadro Optimizasyon Aracı

Veri tabanlı kadro analizi — 1-4-2-3-1 formasyonunda her mevki için en uygun oyuncuyu belirler.

## Nasıl Çalışır?

Her mevki için metrik ağırlıkları ayarlanabilir. Sistem, seçilen ağırlıklara göre normalize edilmiş skorları hesaplar ve en yüksek skoru alan oyuncuyu önerir.

## Metodoloji

- **Normalizasyon:** Min-max (0-1 arası)
- **Ağırlıklandırma:** Wyscout Index & Apunts Journal (2024) referanslı
- **Formasyon:** 1-4-2-3-1

## Veri

FotMob üzerinden derlenen Türkiye A Milli Takımı istatistikleri.

## Kurulum

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Geliştirici

M. Enes Şahin · [menessahin.github.io](https://menessahin.github.io)
