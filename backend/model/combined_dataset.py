import pandas as pd
import datasets

# script for combining the datasets that were used in fine-tuning 

data_1 = pd.read_csv("datasets/phrasebank.csv", encoding='cp1252')
replacement_1 = {"positive": 2, "neutral": 1, "negative": 0}
data_1 = data_1.replace(replacement_1)

data_1.to_csv('datasets/phrasebank.csv', index=False)

#datasets below are loaded from huggingface 

#cleaning datasets 
ds_2 = datasets.load_dataset("lukecarlate/english_finance_news")
data_2 = ds_2['train'].to_pandas()
data_2 = data_2.drop(columns = ['newssource'])
data_2 = data_2.rename(columns={'newscontents': 'text'})
data_2 = data_2[['label', 'text']]

data_2.to_csv('datasets/eng_fin_news.csv', index=False)


ds_3 = datasets.load_dataset("ugursa/Yahoo-Finance-News-Sentences")
data_3 = ds_3['train'].to_pandas()
replacement_2 = {0 : 2, 1 : 1, 2 : 0}

data_3.to_csv('datasets/yahoo_fin_news.csv', index=False)

# concatenates all three datasets and normalizes data

combined_ds = pd.concat([data_1, data_2, data_3], ignore_index=True)
ccomb_ds = combined_ds.drop_duplicates()
ccomb_ds.to_csv("datasets/ccomb_dataset.csv", index=False)
