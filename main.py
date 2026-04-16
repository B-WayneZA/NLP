from src.data_loader import load_language_dataset


## checking if the data loading works
zul_data = load_language_dataset("zulu")

print(zul_data)
print(zul_data["train"][0])