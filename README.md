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
data/raw_data/sephora
```
- Przez robienie projektu na custom dataset należy zmienić nazwę kolumny "rating" z Reviews data na "LABEL-rating"

### 4. Uruchom kontener Dockera z dostępem do środowiska
```bash
make run_docker
```

### 5. Wewnątrz kontenera wykonaj pipeline
```bash
dvc repro
```

### 📁 Struktura konfiguracji

strukturę folderów należy odwzorować w przypadku braków

- `params.yaml` — główny plik parametrów, współdzielony między wszystkimi konfiguracjami. Zawiera ustawienia modelu, cech i ogólne hiperparametry.
- `configs/sephora.yaml`, `configs/rt-polarity.yaml` — konfiguracje specyficzne dla danego zbioru danych (ścieżki, kolumny, parametry pipeline’u).
- `data/raw_data/` — katalog, do którego należy ręcznie wrzucić nieprzetworzone dane wejściowe:
  - `sephora/` z plikami CSV z Kaggle
  - `rt-polarity/` z plikami `rt-polarity.neg` i `rt-polarity.pos`
- `data/processed_data/` — dane wczytane z raw_data, przetworzone na pkl, gotowe do analizy i preprocesingu.
- `data/preprocessed_data/` — dane po wstępnym przetworzeniu.
- `data/train_test_split/` — pliki zawierające podziały na zbiór treningowy i testowy.
- `data/models/` — zapisane modele dla każdej konfiguracji wraz z najlepszymi parametrami w plikach .json.
- `data/metrics/` — metryki ewaluacyjne zapisane dla każdego eksperymentu.
- `data/notebooks/` — dane generowane podczas eksploracji w notatnikach.
- `notebooks/` — eksploracja danych, analizy EDA, szukanie hiperparametrów oraz analiza sharp.
- `excercises/` — katalog na ćwiczenia/testy.
- `results/` — końcowe tabele wyników porównujących modele/datysety.
- `raports/` — wykresy, tabele lub inne artefakty końcowe do raportu.
- `scripts/` — pomocnicze skrypty (np. do pobierania, czyszczenia danych).
- `src/` — główny kod projektu (transformery, funkcje narzędziowe, pipeline).

### 📝 Uwagi
Pipeline automatycznie wykonuje całość przetwarzania i zapisuje wyniki bez potrzeby dodatkowych komend.

Wymagane jest posiadanie konta w Weights & Biases i ustawienie własnego klucza API w .env.

### ✅ Efekt końcowy
Po wykonaniu dvc repro:

- Modele zostaną zapisane w models/{dataset_name}

- Wyniki zostaną zapisane w results/test_results.md

- Szczegółowe metryki w data/metrics/*
