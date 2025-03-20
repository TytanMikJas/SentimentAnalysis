import pandas as pd
import sys
from src.utils import load_raw_data, save_dataset


def process_data(path_to_raw_data, path_to_processed_data):
    reviews_df, product_info_df = load_raw_data(path_to_raw_data)

    sephora_df = pd.merge(
        reviews_df, product_info_df, on="product_id", suffixes=("", "_drop")
    )
    sephora_df = sephora_df.drop(
        columns=[col for col in sephora_df.columns if col.endswith("_drop")]
    )

    # na dobrą sprawę potrzebne jest product_id - identyfikacja, review_text i _title - treść i rating - etykieta
    # że nie robiłem w życiu modelu sentymentu, to niestety nie wiem czy przyda się product_name, brand_name, helpfulness, is_recommended
    # ale zapisuję je na wszelki wypadek jakbyśmy chcieli kontekstowo analizować dane
    cols_to_save = [
        "product_id",
        "product_name",
        "brand_name",
        "helpfulness",
        "review_title",
        "review_text",
        "LABEL-rating",
    ]

    save_dataset(sephora_df[cols_to_save], path_to_processed_data)


if __name__ == "__main__":
    path_to_raw_data = sys.argv[1]
    path_to_processed_data = sys.argv[2]
    process_data(path_to_raw_data, path_to_processed_data)
