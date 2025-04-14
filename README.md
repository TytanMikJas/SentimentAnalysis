# Sentiment Analysis Pipeline

Ten projekt zawiera kompletny pipeline eksperymentów z użyciem `DVC`, `Docker` i `wandb`. Instrukcja poniżej pozwoli Ci zbudować środowisko i zreprodukować wyniki.

---

## 🐳 Uruchomienie przez Dockera

### 1. Zbuduj obraz Dockera

```bash
make build
```

### 2. Stwórz plik .env z kluczem API do Weights & Biases

WANDB_API_KEY=twoj_klucz_api

### 3. Pobierz dane i rozmieść je w katalogach
🗂️ rt-polarity

- Pobierz dane z: http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz

- Wypakuj zawartość do:
```bash
data/raw_data/rt-polarity/
```

- Tak aby w środku były pliki:
```
rt-polarity.neg
rt-polarity.pos
```

🗂️ sephora
- Pobierz dane z: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews

- Wypakuj wszystkie pliki .csv do:

```bash
data/raw_data/sephora/
```

### 4. Uruchom kontener Dockera z dostępem do środowiska
```bash
make run_docker
```

### 5. Wewnątrz kontenera wykonaj pipeline
```bash
dvc repro
```

### 📁 Struktura konfiguracji

- params.yaml — główny plik parametrów, współdzielony między konfiguracjami.

- configs/sephora.yaml, configs/rt-polarity.yaml — konfiguracje specyficzne dla zbiorów danych.

- models/ — katalog z zapisanymi najlepszymi modelami (dla każdego datasetu osobno).

- data/metrics/ — zawiera metryki dla każdego z eksperymentów.

- results/ — wygenerowane podsumowania wyników.

### 📝 Uwagi
Pipeline automatycznie wykonuje całość przetwarzania i zapisuje wyniki bez potrzeby dodatkowych komend.

Wymagane jest posiadanie konta w Weights & Biases i ustawienie własnego klucza API w .env.

### ✅ Efekt końcowy
Po wykonaniu dvc repro:

- Modele zostaną zapisane w models/{dataset_name}

- Wyniki zostaną zapisane w results/test_results.md oraz .pdf

- Szczegółowe metryki w data/metrics/*