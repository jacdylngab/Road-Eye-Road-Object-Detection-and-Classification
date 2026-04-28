import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("Training vs. Validation Loss_Training.csv")
val_df = pd.read_csv("Training vs. Validation Loss_Validation.csv")

train_df.columns = ["wall_time", "step", "train_loss"]
val_df.columns = ["wall_time", "step", "val_loss"]

train_df = train_df[["step", "train_loss"]]
val_df = val_df[["step", "val_loss"]]

df = pd.merge(train_df, val_df, on="step", how="outer")
df = df.sort_values("step")

plt.plot(df["step"], df["train_loss"], label="Train Loss")
plt.plot(df["step"], df["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("train_vs_val.png")