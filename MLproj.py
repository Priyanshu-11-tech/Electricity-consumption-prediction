import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Global variables
data = None
X = None
y = None
future_data = None

# Functions
def load_dataset():
    global data
    file_path = filedialog.askopenfilename(title="Select Dataset File", filetypes=(("CSV Files", "*.csv"),))
    if file_path:
        try:
            data = pd.read_csv(file_path)
            preprocess_data()
            messagebox.showinfo("Success", "Dataset loaded and preprocessed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    else:
        messagebox.showwarning("Warning", "No file selected.")

def preprocess_data():
    global X, y
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Weekday'] = data['Date'].dt.weekday
    data.fillna(method='ffill', inplace=True)
    data['Temperature^2'] = data['Temperature'] ** 2
    X = data[['Temperature', 'Temperature^2', 'Day', 'Month', 'Year', 'Weekday']]
    y = data['Consumption_kWh']

def train_model():
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        messagebox.showinfo("Training Complete", f"Model trained successfully!\nMSE: {mse:.2f}\nR2 Score: {r2:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")

def predict_future():
    global future_data
    try:
        start_date = start_date_entry.get()  # Get the start date
        num_days = int(days_entry.get())  # Number of days to predict
        tariff_rate = float(tariff_entry.get())  # Tariff rate input

        # Calculate average temperature from past data
        avg_temp = data['Temperature'].mean()  # Get the mean temperature from the dataset

        # Generate future dates
        future_dates = pd.date_range(start=start_date, periods=num_days, freq='D')

        # Prepare the future data
        future_data = pd.DataFrame({
            'Date': future_dates,
            'Temperature': avg_temp,
            'Temperature^2': avg_temp ** 2,
            'Day': future_dates.day,
            'Month': future_dates.month,
            'Year': future_dates.year,
            'Weekday': future_dates.weekday
        })

        # Predict future consumption using the trained model
        future_data['Predicted_Consumption_kWh'] = model.predict(future_data[['Temperature', 'Temperature^2', 'Day', 'Month', 'Year', 'Weekday']])
        
        # Calculate predicted bill based on tariff rate
        future_data['Predicted_Bill_₹'] = future_data['Predicted_Consumption_kWh'] * tariff_rate
        
        # Calculate total bill
        total_bill = future_data['Predicted_Bill_₹'].sum()

        # Display the total bill
        result_text.set(f"Total Predicted Bill: ₹{total_bill:.2f}")
        
        # Save predictions to CSV
        future_data.to_csv("future_predictions_with_bill.csv", index=False)
        messagebox.showinfo("Success", "Predictions saved to 'future_predictions_with_bill.csv'!")

        # Plot the future predictions
        plot_future_predictions()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict future consumption: {e}")


def plot_past_consumption():
    try:
        ax1.clear()
        ax1.plot(data['Date'], data['Consumption_kWh'], label="Past Consumption (kWh)", color='blue', marker='o')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Electricity Consumption (kWh)")
        ax1 .set_title("Past Electricity Consumption")
        ax1.legend()
        ax1.grid()
        canvas1.draw()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot past consumption: {e}")

def plot_future_predictions():
    try:
        ax2.clear()
        ax2.plot(future_data['Date'], future_data['Predicted_Consumption_kWh'], label="Predicted Consumption (kWh)", color='orange', marker='o')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Electricity Consumption (kWh)")
        ax2.set_title("Predicted Electricity Consumption for Future Dates")
        ax2.legend()
        ax2.grid()
        canvas2.draw()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot future predictions: {e}")

# GUI Layout
root = tk.Tk()
root.title("Electricity Consumption Prediction")
root.geometry("800x700")
root.configure(bg="#e0f7fa")

# Frames for better organization
header_frame = tk.Frame(root, bg="#4CAF50")
header_frame.pack(fill=tk.X)

title_label = tk.Label(header_frame, text="Electricity Consumption Prediction", font=("Arial", 20, "bold"), bg="#4CAF50", fg="white")
title_label.pack(pady=10)

input_frame = tk.Frame(root, bg="#e0f7fa")
input_frame.pack(pady=10)

load_button = tk.Button(input_frame, text="Load Dataset", command=load_dataset, width=20, bg="#2196F3", fg="white", font=("Arial", 12))
load_button.grid(row=0, column=0, padx=5, pady=5)

train_button = tk.Button(input_frame, text="Train Model", command=train_model, width=20, bg="#FF9800", fg="white", font=("Arial", 12))
train_button.grid(row=0, column=1, padx=5, pady=5)

start_date_label = tk.Label(input_frame, text="Start Date (YYYY-MM-DD):", bg="#e0f7fa", font=("Arial", 12))
start_date_label.grid(row=1, column=0, sticky=tk.W, padx=5)

start_date_entry = tk.Entry(input_frame, font=("Arial", 12))
start_date_entry.grid(row=1, column=1, padx=5)

days_label = tk.Label(input_frame, text="Number of Days:", bg="#e0f7fa", font=("Arial", 12))
days_label.grid(row=2, column=0, sticky=tk.W, padx=5)

days_entry = tk.Entry(input_frame, font=("Arial", 12))
days_entry.grid(row=2, column=1, padx=5)

temp_label = tk.Label(input_frame, text="Average Temperature (°C):", bg="#e0f7fa", font=("Arial", 12))
temp_label.grid(row=3, column=0, sticky=tk.W, padx=5)

temp_entry = tk.Entry(input_frame, font=("Arial", 12))
temp_entry.grid(row=3, column=1, padx=5)

tariff_label = tk.Label(input_frame, text="Tariff Rate (₹/kWh):", bg="#e0f7fa", font=("Arial", 12))
tariff_label.grid(row=4, column=0, sticky=tk.W, padx=5)

tariff_entry = tk.Entry(input_frame, font=("Arial", 12))
tariff_entry.grid(row=4, column=1, padx=5)

predict_button = tk.Button(input_frame, text="Predict Future Consumption", command=predict_future, width=25, bg="#FF5722", fg="white", font=("Arial", 12))
predict_button.grid(row=5, columnspan=2, pady=10)

plot_past_button = tk.Button(input_frame, text="Plot Past Consumption", command=plot_past_consumption, width=25, bg="#3F51B5", fg="white", font=("Arial", 12))
plot_past_button.grid(row=6, columnspan=2, pady=5)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14), fg="green", bg="#e0f7fa")
result_label.pack(pady=10)

# Matplotlib Figures and Canvases
fig1, ax1 = plt.subplots(figsize=(6, 4))
canvas1 = FigureCanvasTkAgg(fig1, master=root)
canvas_widget1 = canvas1.get_tk_widget()
canvas_widget1.pack(pady=10)

toolbar1 = NavigationToolbar2Tk(canvas1, root)
toolbar1.update()
canvas1.get_tk_widget().pack()

fig2, ax2 = plt.subplots(figsize=(6, 4))
canvas2 = FigureCanvasTkAgg(fig2, master=root)
canvas_widget2 = canvas2.get_tk_widget()
canvas_widget2.pack(pady=10)

toolbar2 = NavigationToolbar2Tk(canvas2, root)
toolbar2.update()
canvas2.get_tk_widget().pack()

# Run the app
root.mainloop()