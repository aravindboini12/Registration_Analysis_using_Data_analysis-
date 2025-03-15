# Vehicle Registration Analytics Dashboard

This project is a comprehensive Streamlit dashboard for vehicle registration analysis, prediction, and recommendations. It includes visualizations, machine learning models, and a personalized vehicle recommendation system.

## Features

- Number of Vehicles Registered by Location
- Fuel Type Distribution Visualization
- EV Vehicle Count Display
- Predict Insurance Validity and Fuel Type
- Vehicle Recommendation Based on User Input and Location
- Check Vehicle Details by Registration Number (fuel type, model, body type, engine capacity, location)

---

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/aravindboini12/Registration_Analysis_using_Data_analysis-
    ```
2. Navigate to the project folder:
    ```bash
    cd Registration_Analysis_using_Data_analysis-
    ```
3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit dashboard:
    ```bash
    streamlit run app.py
    ```

---

## Dataset Format
Ensure your dataset (`DATASET.csv`) follows this format:

| registrationNo | fuel | modelDesc | bodyType | cc | hp | seatCapacity | cylinder | OfficeCd | regvalidfrom | regvalidto | fromdate | todate |
|----------------|------|-----------|----------|----|----|--------------|----------|----------|--------------|------------|----------|--------|
| TG343165       | Petrol | Swift     | Hatchback | 1197 | 81 | 5            | 4        | ABC123   | 01/01/2020  | 01/01/2025 | 01/01/2020 | 01/01/2024 |

---

## Machine Learning Models

- Linear Regression - Predicts insurance validity based on vehicle details
- Random Forest Classifier - Predicts fuel type based on vehicle specifications

---

## How to Use

1. Select Fuel Type in the sidebar filter
2. Input Vehicle Details (cc, hp, seat capacity, cylinders, and location)
3. Press 'Predict' to get:
   - Predicted Insurance Validity Date
   - Predicted Fuel Type
   - Recommended Vehicles (based on location and input details)
4. Enter Registration Number to fetch existing vehicle details
5. EV Recommendations appear if the selected location has more than 50 EVs

---

## Error Handling & Troubleshooting

- If dates don't parse correctly, ensure the date format is `DD/MM/YYYY`
- If no vehicle data appears, double-check the registration number format
- If the app crashes, restart with `streamlit run app.py`

---

## Future Enhancements

- More ML models for performance comparison
- Enhanced recommendations with user reviews and pricing data
- Geolocation-based recommendations

---

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to modify.

# for any type of queries  feel free to mail :aravindboini1225@gmail.com

Happy analyzing!

