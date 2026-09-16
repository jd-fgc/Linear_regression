import argparse
import sys


def do_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear Regression")
    p.add_argument("--data", required=True, help="Name of the data file")
    args = p.parse_args()
    return args


def read_data(file: str) -> str:
    with open(file, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def format_data(data: str):
    km_data = []
    price_data = []
    lines = data.split("\n")

    for line in lines[1:]:
        if line:
            data_format = line.split(',')
            km, price = float(data_format[0]), float(data_format[1])
            km_data.append(km)
            price_data.append(price)
    return km_data, price_data


def normalized(value: list):
    min_v = min(value)
    max_v = max(value)
    normalized = []

    for v in value:
        normalized.append((v - min_v) / (max_v - min_v))
    return normalized, min_v, max_v


def gradient(km_norm, price_norm, learning_rate, iterations):
    theta0 = 0
    theta1 = 0
    m = len(km_norm)

    for _ in range(iterations):
        error_sum0 = 0
        error_sum1 = 0

        for i in range(m):
            prediction = theta0 + theta1 * km_norm[i]
            error = prediction - price_norm[i]
            error_sum0 = error_sum0 + error
            error_sum1 = error_sum1 + error * km_norm[i]

        tmp_theta0 = learning_rate * (1/m) * error_sum0
        tmp_theta1 = learning_rate * (1/m) * error_sum1

        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1

    return theta0, theta1


def denormalize(theta0_norm, theta1_norm, km_min, km_max, price_min, price_max):
    theta1 = (price_max - price_min) * theta1_norm / (km_max - km_min)
    theta0 = price_min + (price_max - price_min) * theta0_norm - theta1 * km_min
    return theta0, theta1


def save_theta(theta0, theta1, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(theta0) + "\n" + str(theta1))


def main():
    try:
        args = do_args()
        file = args.data
        data = read_data(file)
        km, price = format_data(data)
        km_norm, km_min, km_max = normalized(km)
        price_norm, price_min, price_max = normalized(price)
        theta0, theta1 = gradient(km_norm, price_norm, 0.2, 2000)
        theta0, theta1 = denormalize(theta0, theta1, km_min, km_max, price_min, price_max)
        save_theta(theta0, theta1, "data/save.csv")
    except Exception as e:
        print(f"Error : {e}")
        sys.exit()


if __name__ == "__main__":
    main()
