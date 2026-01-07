import pandas as pd

df1 = pd.read_csv("valid_solutions.csv")
df2 = pd.read_csv("valid_guesses.csv")
merged = pd.concat([df1, df2]).drop_duplicates()
merged.to_csv("merged.csv", index=False)

print(len(merged))
