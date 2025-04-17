# Sentiment Analysis Pipeline
This project contains a complete pipeline of experiments using `DVC`, `Docker` and `wandb` . The instructions below will allow you to build the environment and reproduce the results.

---
## 🐳 Run via Docker
### 1. Build a Docker image
``bash
make build
```
### 2. Create an .env file with the API key for Weights & Biases
WANDB_API_KEY=your_key_api
### 3. Download the data and deploy it to the
🗂️ rt-polarity
- Download the data from: http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
- Extract the contents to:
```bash
data/raw_data/rt-polarity/
```
- So that the files inside are:
```
rt-polarity.neg
rt-polarity.pos
```
🗂️ sephora
- Download the data from: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews
- Extract all .csv files to:
```bash
data/raw_data/sephora
```
- By doing the project on custom dataset, rename the "rating" column from Reviews data to "LABEL-rating".
### 4. Start a Docker container with access to the environment
```bash
make run_docker
```
### 5. Inside the container, execute the pipeline
```bash
dvc repro
```
### 📁 Configuration structure
folder structure should be mapped in case of deficiencies
- `params.yaml` - main parameter file, shared among all configurations. Contains model settings, features and general hyperparameters.
- `configs/sephora.yaml`, `configs/rt-polarity.yaml` - dataset-specific configurations (paths, columns, pipeline parameters).
- `data/raw_data/` - directory where you should manually drop the raw input data into:
- `sephora/` with CSV files from Kaggle
- `rt-polarity/` with files `rt-polarity.neg` and `rt-polarity.pos`
- `data/processed_data/` - data loaded from raw_data, processed into pkl, ready for analysis and preprocessing.
- `data/preprocessed_data/` - data after preprocessing.
- `data/train_test_split/` - files containing splits for training and test collection.
- `data/models/` - saved models for each configuration along with the best parameters in .json files.
- `data/metrics/` - evaluation metrics saved for each experiment.
- `data/notebooks/` - data generated during exploration in notebooks.
- `notebooks/` - data mining, EDA analysis, hyperparameter search and sharp analysis.
- `excercises/` - directory for exercises/tests.
- `results/` - final tables of results comparing models/datasets.
- `reports/` - graphs, tables or other final artifacts for the report.
- `scripts/` - auxiliary scripts (e.g. for downloading, cleaning data).
- `src/` - main project code (transformers, tool functions, pipeline).
- 
### 📝 Notes
Pipeline automatically performs all processing and saves the results without the need for additional commands.
It is required to have an account in Weights & Biases and set your own API key in .env.

### ✅ End result
After dvc repro:
- Models will be saved in models/{dataset_name}
- Results will be saved in results/test_results.md
- Detailed metrics in data/metrics/*