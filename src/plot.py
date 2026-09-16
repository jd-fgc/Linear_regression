import argparse
import matplotlib.pyplot as plt

from train import read_data, format_data
from predict import predict


def do_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear Regression - Plot")
    p.add_argument("--data", required=True, help="Name of the data file")
    p.add_argument("--save", required=True, help="Name of the save theta file")
    args = p.parse_args()
    return args


def load_theta(data: str):
    lines = data.split("\n")
    theta0 = float(lines[0])
    theta1 = float(lines[1])
    return theta0, theta1


def plot_regression(km, price, theta0, theta1):
    plt.scatter(km, price, color="blue", label="Data")

    km_line = [min(km), max(km)]
    price_line = [predict(theta0, theta1, km_line[0]),
                  predict(theta0, theta1, km_line[1])]
    plt.plot(km_line, price_line, color="red", label="Regression line")

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Linear Regression - Price vs Mileage")
    plt.legend()
    plt.show()


def main():
    try:
        args = do_args()
        data = read_data(args.data)
        theta_data = read_data(args.save)

        km, price = format_data(data)
        theta0, theta1 = load_theta(theta_data)

        plot_regression(km, price, theta0, theta1)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
