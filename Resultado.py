import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("prediccion.csv")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")

    plt.figure(figsize=(12, 6))
    plt.plot(df["fecha"], df["precio"], label="Precio real", color="black")
    plt.plot(df["fecha"], df["predicho"], label="Precio predicho", color="blue")
    plt.title("Grafica: Precio Real vs Predicho")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()