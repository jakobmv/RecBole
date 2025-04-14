from recbole.quick_start import run_recbole

if __name__ == "__main__":
    # Run the model with yaml configuration
    run_recbole(
        model="DESSBERT4Rec",
        dataset="ml-100k-wiki",
        config_file_list=[
            "recbole/properties/dataset/ml-100k-wiki.yaml",
            "recbole/properties/model/DESSBERT4Rec.yaml",
        ],
    )
