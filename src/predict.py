import argparse
import sys


def do_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear Regression")
    p.add_argument("--data", required=False, help="Name of the data file")
    args = p.parse_args()
    return args


def read_data(file: str) -> str:
    with open(file, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def predict(theta0, theta1, mileage):
    estimate = theta0 + theta1 * mileage
    return estimate


def main():
    try:
        args = do_args()
        file = args.data

        theta0 = 0
        theta1 = 0
        if file is not None:
            try:
                data = read_data(file)
                data_split = data.split('\n')
                theta0 = float(data_split[0])
                theta1 = float(data_split[1])
            except FileNotFoundError:
                print("File not find, Theta init to 0")
        mileage = float(input("Enter mileage: "))
        final_price = predict(theta0, theta1, mileage)
        print(f"Predict price: {round(final_price, 2)}$")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit()


if __name__ == "__main__":
    main()
