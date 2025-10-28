# NexGen RoutePrime: Smart Route Planner ✨

**Author:** [Your Name]
**Case Study:** OFI AI Internship - Logistics Innovation Challenge
**Option Chosen:** 2 - Smart Route Planner

---

## 1. Project Overview 🚀

**NexGen RoutePrime** is an interactive web application built with Streamlit designed to solve the "Smart Route Planner" challenge. This tool empowers a logistics manager at NexGen to optimize delivery routes by providing a data-driven, prescriptive solution.

**Key Features:**
* **Warehouse & Vehicle Selection:** Allows users to choose an origin warehouse (dynamically populated from available pending orders) and an available vehicle from the fleet.
* **Order Selection:** Dynamically filters and displays pending orders for the selected warehouse, allowing users to choose which ones to include in the route.
* **Multi-Objective Optimization:** Enables users to define the primary optimization goal: minimizing **Cost**, **Time**, or **Environmental Impact (CO2)**.
* **Advanced Solver:** Utilizes Google's **OR-Tools** library to solve the Capacitated Vehicle Routing Problem (CVRP), finding the most efficient sequence of stops based on the chosen objective.
* **Interactive Visualization:** Displays the optimized route on an interactive **Folium map** with clear markers and path.
* **KPI Analysis:** Presents key performance indicators (KPIs) for the generated route, highlighting the **specific metric that was optimized** using the solver's objective value, alongside the calculated totals for other metrics along that route.
* **Data Export:** Allows users to **download** the detailed stop-by-stop route plan as a CSV file.

This application directly addresses NexGen's challenges of **Operational Inefficiencies**, **Cost Pressures**, and the need for **Innovation**, informed by the initial data analysis performed.

---

## 2. Exploratory Data Analysis (EDA) 📊

Prior to building the application, an **Exploratory Data Analysis (EDA)** was conducted using the `Main_Work.ipynb` Jupyter Notebook to gain insights from the provided datasets. This crucial step involved:

* **Loading & Merging:** Reading all 7 CSV files into pandas DataFrames and integrating `orders` with `delivery_performance` data.
* **Data Cleaning:** Handling missing values, specifically:
    * Identifying pending orders by checking for null values in the `Delivery_Status` column.
    * Filling missing `quality_issue` entries with the string 'None' for proper categorical analysis.
* **Robust Column Handling:** Implementing a helper function (`find_column_name`) to dynamically detect the correct column names despite potential variations in capitalization or spacing (e.g., finding `Order_ID`, `origin`, `Delivery_Status`, `promised_delivery_days`, `actual_delivery_days`, `quality_issue`, cost components, etc.).
* **Feature Engineering:** Creating valuable derived metrics for analysis:
    * `On-Time Status`: Comparing actual vs. promised delivery durations (days).
    * `Delivery Overrun (days)`: Calculating the difference between actual and promised days to quantify delays.
    * `Total Order Cost`: Aggregating various cost components from `cost_breakdown.csv`.
* **Insight Generation (`print_insights`):** Identifying and printing key findings:
    * Quantified the overall on-time delivery rate and average delay duration.
    * Attempted to correlate delays with customer ratings (limited by missing rating data in the log).
    * Identified the top quality issues (excluding 'None').
    * Calculated the number and estimated value of pending orders.
    * Analyzed the average `Total Order Cost` per delivery `Priority`.
* **Visualization (`visualize_data`):** Generating plots using **Matplotlib and Seaborn** to illustrate the findings:
    * **Pie Chart:** Showing the proportion of On-Time vs. Delayed deliveries.
    * **Bar Chart:** Displaying the Average Total Order Cost by Priority.
    * **Histogram:** Visualizing the distribution of Delivery Overrun days for delayed orders.
    * **Count Plot:** Showing the frequency of different Quality Issues (excluding 'None').

The EDA provided a foundational understanding of data quality, key performance indicators, and existing operational challenges, directly informing the requirements and focus of the **NexGen RoutePrime** Streamlit application.

---

## 3. Fulfillment of Technical Requirements ✔️

This project meticulously addresses all technical requirements specified in the case study brief:

* **Python and Streamlit:**
    * ✅ **Python:** All data loading, EDA (in `.ipynb`), analysis, optimization, and application logic (`.py`) are written exclusively in Python.
    * ✅ **Streamlit:** An interactive web application (`your_app.py`) is built using the Streamlit framework.
    * ✅ **Local Execution:** The application runs locally using `streamlit run your_app.py`.

* **Data Analysis:**
    * ✅ **Multiple Datasets:** Both the EDA notebook and the Streamlit app load and integrate data from the relevant CSV files.
    * ✅ **Meaningful Calculations:** The EDA calculates KPIs like on-time rate and average costs. The app calculates simulated travel matrices (`create_cost_matrix`) based on data averages and assumptions, forming the core constraints for the optimizer.
    * ✅ **Handle Missing Data:** Pending orders are identified via null status in both EDA and app. Missing quality issues are handled. The app gracefully handles potential missing columns during key finding.
    * ✅ **Derived Metrics:** EDA creates `On-Time Status`, `Delivery Overrun`, `Total Order Cost`. The App creates `Demand` (simulated), `Capacity` (mapped), and the crucial `cost`, `time`, `co2` **matrices** needed for optimization.

* **Visualization:**
    * ✅ **4+ Chart Types:** The EDA notebook (`Main_Work.ipynb`) generates **Pie, Bar, Histogram, and Countplot** using Matplotlib/Seaborn.
    * ✅ **Interactive Visualization:** The Streamlit application features an interactive **Folium map** for route visualization (pan, zoom, markers).
    * ✅ **Appropriate Charts:** The Folium map is perfectly suited for displaying routes. Charts in the EDA notebook were chosen appropriately (Pie for share, Bar for category comparison, Histogram for distribution, Countplot for frequency).

* **Interactivity:**
    * ✅ **Filters/Selections/Inputs:** The Streamlit sidebar uses `selectbox` for warehouse (data-driven) and vehicle, `multiselect` for orders, and `selectbox` for optimization goal (`Cost`/`Time`/`Environmental Impact`).
    * ✅ **Dynamic Response:** The main page (map, KPIs, route plan) updates after clicking the "Generate Optimized Route" button. Results are stored in `st.session_state` and displayed persistently using `st.rerun()`. The order list updates based on the warehouse selection.
    * ✅ **Download/Export:** A styled download link allows exporting the generated `route_plan_df` as a CSV file.

* **Code Quality:**
    * ✅ **Well-Organized/Readable:** Code in both the notebook and app is structured into functions with clear names. The app uses logical sections for imports, config, data loading, helpers, solver, and UI.
    * ✅ **Comments/Documentation:** Functions have docstrings. Inline comments explain assumptions, logic steps, and fixes.
    * ✅ **Proper Error Handling:** Includes `try-except` for file loading and solver execution. Uses `st.error`, `st.warning`, `st.info` for user feedback. Explicit checks for missing critical columns and unmet constraints (e.g., capacity) use `st.stop()` to prevent crashes.
    * ✅ **Efficient Data Processing:** Uses pandas for vectorized operations. Leverages Streamlit's `@st.cache_data` on key functions (`load_all_data`, `get_mock_geodata`, `create_cost_matrix`) to minimize re-computation during user interactions.

* **Bonus: Advanced Features (+20):**
    * ✅ **Optimization:** Explicitly utilizes Google **OR-Tools** to model and solve the Capacitated Vehicle Routing Problem (CVRP), providing a mathematically optimal solution based on the chosen objective. This fulfills the "optimization" criterion.

---

## 4. How to Run ⚙️

### 4.1. Prerequisites

* Python 3.8+ installed.
* Access to a terminal or command prompt.
* All 7 dataset CSV files provided in the case study brief.
* (Optional) Jupyter Notebook environment to run the EDA.

### 4.2. Setup

1.  **Folder Structure:** Organize your project as follows:
    ```
    your_project_folder/
    ├── data/
    │   ├── orders.csv
    │   ├── delivery_performance.csv
    │   ├── routes_distance.csv
    │   ├── vehicle_fleet.csv
    │   ├── cost_breakdown.csv
    │   ├── warehouse_inventory.csv  (Optional, not used by final app)
    │   └── customer_feedback.csv    (Optional, not used by final app)
    ├── your_app.py                # The Streamlit application
    ├── Main_Work.ipynb            # Your Jupyter Notebook for EDA
    └── requirements.txt           # Python libraries needed
    ```
2.  **CSV Files:** Place the required CSV files inside the `data/` subfolder.

### 4.3. Installation

1.  **Create `requirements.txt`:** Ensure this file exists in `your_project_folder/` with the content:
    ```text
    # For Streamlit App
    streamlit
    pandas
    numpy
    folium
    streamlit-folium
    ortools

    # For EDA Notebook
    matplotlib
    seaborn
    notebook
    ```
2.  **Install Libraries:** Open your terminal, navigate (`cd`) to `your_project_folder/`, and run:
    ```bash
    pip install -r requirements.txt
    ```

### 4.4. Launch the Application

1.  **Navigate:** Ensure your terminal is in `your_project_folder/`.
2.  **Run:** Execute:
    ```bash
    streamlit run your_app.py
    ```

Your browser should open the **NexGen RoutePrime** application.

---

## 5. Key Assumptions & Limitations ⚠️

* **Mock Geodata:** Latitude/Longitude coordinates were **simulated**.
* **Mock Order Demand:** Order size/volume was **simulated**.
* **Simplified Vehicle Capacity:** Vehicle types were mapped to numeric `Capacity`.
* **Simulated Cost/Time/CO2 Matrix:** Travel metrics were **simulated** based on averages and mock distances, with mathematical adjustments (v10+) to encourage differentiation between optimization goals.
* **Simulated Fuel Price:** A mock fuel price (`95 INR/L`) was used.

### Limitation: Dataset Size & Simulation Correlation

* Due to the **small dataset** size and the **simulated nature** of cost factors (even with adjustments), the optimizer might still produce **similar routes** for different goals (Cost, Time, CO2). The underlying "best path" considering capacity and stops might remain largely the same across the simulated metrics.
* With a larger, real-world dataset featuring genuine, complex trade-offs (e.g., toll roads vs. local roads, time-varying traffic, specific vehicle efficiencies), the distinct optimization goals are expected to yield more significantly different route recommendations and KPI outcomes.