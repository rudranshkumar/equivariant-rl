import glob, json
import pandas as pd

rows = [json.load(open(p)) for p in glob.glob("results/*.json")]
df = pd.json_normalize(rows)

df = df.drop_duplicates(subset=["trial_uid"])


df.to_csv("refine_results.csv", index=False)

print("N =", len(df))
print(df.sort_values("score", ascending=False).head(10)[["exp","score","seed","time_sec"]])
print(df.sort_values("score", ascending=False).head(10)[["exp","score","seed","time_sec"]])
print(df.groupby("exp")["score"].describe()[["count","mean","std","max"]])

