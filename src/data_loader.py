from datasets import load_dataset


def load_language_dataset(language_code):
   dataset = load_dataset("parquet", data_files={
        "train": f"data/{language_code}/train.parquet",
        "validation": f"data/{language_code}/dev.parquet",
        "test": f"data/{language_code}/test.parquet",
   })

   return dataset


def load_all_languages():
   """
   Loads all languages into a dictionary
   """
   languages = ["zul", "xho", "swa"]
   
   datasets = {}
   for lang in languages:
      datasets[lang] = load_language_dataset(lang)
   
   return datasets