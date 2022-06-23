import pandas as pd
from sklearn.model_selection import train_test_split

combined_metadata = "raw_metadata.csv"
train_metadata = "train_metadata.csv"
test_metadata = "test_metadata.csv"

df = pd.read_csv(combined_metadata)
train, test = train_test_split(df,test_size=0.2)

train.to_csv(train_metadata,index=False)
test.to_csv(test_metadata,index=False)