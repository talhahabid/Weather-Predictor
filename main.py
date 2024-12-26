import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

weather = pd.read_csv("data.csv", index_col="DATE")
core_weather = weather[["PRCP", "SNWD", "TMAX", "TMIN"]].copy()
core_weather.columns = ["precip", "snow_depth", "temp_max", "temp_min"]

# Cleaning Data
core_weather.drop(columns=["snow_depth"], inplace=True)
core_weather["precip"] = core_weather["precip"].fillna(0)
core_weather.fillna(method="ffill", inplace=True)

core_weather.index = pd.to_datetime(core_weather.index)

core_weather["target"] = core_weather["temp_max"].shift(-1)
core_weather = core_weather.iloc[:-1, :].copy()

core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()
core_weather["month_day_max_ratio"] = core_weather["month_max"] / core_weather["temp_max"]
core_weather["max_min_ratio"] = core_weather["temp_max"] / core_weather["temp_min"]

core_weather = core_weather.iloc[30:].copy()
core_weather = core_weather[core_weather["temp_min"] != 0]

core_weather["monthly_avg"] = core_weather.groupby(core_weather.index.month)["temp_max"].transform(lambda x: x.expanding().mean())
core_weather["day_of_year_avg"] = core_weather.groupby(core_weather.index.day_of_year)["temp_max"].transform(lambda x: x.expanding().mean())

predictors = ["precip", "temp_max", "temp_min", "month_max", "month_day_max_ratio", "max_min_ratio", "day_of_year_avg", "monthly_avg"]

def create_prediction(predictors, data, model):
    train = data.loc[:"2011-12-31"]  
    test = data.loc["2012-01-01":]
    model.fit(train[predictors], train["target"])
    predictions = model.predict(test[predictors])
    mae = mean_absolute_error(test["target"], predictions)
    mse = ((test["target"] - predictions) ** 2).mean()
    r2 = model.score(test[predictors], test["target"])
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return mae, mse, r2, combined


reg = Ridge(alpha=0.1)
final_mae, final_mse, final_r2, final_combined = create_prediction(predictors, core_weather, reg)

def plot_metrics(mae, mse, r2):
    metrics = {"Mean Absolute Error": mae, "Mean Squared Error": mse, "R-squared": r2}
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green'])
    plt.title("Model Performance Metrics")
    plt.ylabel("Metric Value")
    plt.show()

def menu():
    while True:
        print("\nMenu:")
        print("1. Predict next day's temperature based on custom input")
        print("2. Plot actual vs predicted temperatures")
        print("3. Display model coefficients")
        print("4. Show Mean Absolute Error of model")
        print("5. Show performance metrics and plot")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            try:
                precip = float(input("Enter precipitation (in mm): "))
                temp_max = float(input("Enter today's max temperature (°F): "))
                temp_min = float(input("Enter today's min temperature (°F): "))
                month_max = temp_max
                month_day_max_ratio = month_max / temp_max
                max_min_ratio = temp_max / temp_min
                monthly_avg = temp_max
                day_of_year_avg = temp_max

                input_data = pd.DataFrame([[precip, temp_max, temp_min, month_max, month_day_max_ratio, max_min_ratio, day_of_year_avg, monthly_avg]], columns=predictors)

                prediction = reg.predict(input_data)[0]
                print(f"Predicted max temperature for next day: {prediction:.2f}°F")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == "2":
            final_combined.plot(title="Actual vs Predicted Temperatures", figsize=(10, 6))
            plt.xlabel("Date")
            plt.ylabel("Temperature (°F)")
            plt.show()

        elif choice == "3":
            print("Model Coefficients:")
            for predictor, coef in zip(predictors, reg.coef_):
                print(f"{predictor}: {coef:.4f}")

        elif choice == "4":
            print(f"Mean Absolute Error (MAE) of the model: {final_mae:.2f}")

        elif choice == "5":
            print("Performance Metrics:")
            print(f"Mean Absolute Error (MAE): {final_mae:.2f}")
            print(f"Mean Squared Error (MSE): {final_mse:.2f}")
            print(f"R-squared (R2): {final_r2:.2f}")
            plot_metrics(final_mae, final_mse, final_r2)

        elif choice == "6":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

menu()