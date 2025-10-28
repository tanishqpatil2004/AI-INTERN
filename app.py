"""
NexGen RoutePrime: Smart Route Planner (OFI Case Study)

(Version 13: Displays the direct objective value from the solver
 to better highlight the specific metric optimized, alongside the
 calculated totals for other metrics on that route.)
"""

# --- 1. Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
from streamlit_folium import st_folium
import base64
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="NexGen RoutePrime", page_icon="🚚", layout="wide", initial_sidebar_state="expanded"
)

# --- 3. Data Loading & Caching (Keep functions as they were) ---
# ... (load_all_data, find_column_name, get_mock_geodata, create_cost_matrix functions remain the same) ...
# --- Re-add functions here for completeness ---
def find_column_name(df_columns: list, potential_names: list) -> str:
    """Helper function to find column names."""
    for name in potential_names:
        if name in df_columns: return name
    return None

@st.cache_data
def load_all_data(data_path="data/"):
    """Loads necessary CSV files and processes data."""
    print("--- Loading all data... ---")
    datasets = ["orders.csv", "delivery_performance.csv", "routes_distance.csv", "vehicle_fleet.csv", "cost_breakdown.csv"]
    data_frames = {}
    try:
        for file in datasets:
            file_path = os.path.join(data_path, file); df_name = file.split('.')[0]
            data_frames[df_name] = pd.read_csv(file_path)
    except FileNotFoundError as e:
        st.error(f"❌ Error: Data file not found: {e.filename}. Check 'data' folder."); return None, None, None, None

    keys = {}
    try:
        keys["order_id"] = find_column_name(data_frames["orders"].columns, ["Order_ID", "order_id"])
        keys["delivery_status"] = find_column_name(data_frames["delivery_performance"].columns, ["Delivery_Status", "delivery_status"])
        keys["origin"] = find_column_name(data_frames["orders"].columns, ["Origin", "origin"])
        keys["destination"] = find_column_name(data_frames["orders"].columns, ["Destination", "destination"])
        keys["vehicle_id"] = find_column_name(data_frames["vehicle_fleet"].columns, ["Vehicle_ID", "Vehicle ID"])
        keys["vehicle_type"] = find_column_name(data_frames["vehicle_fleet"].columns, ["Vehicle_Type", "Vehicle Type"])
        keys["co2_kg_km"] = find_column_name(data_frames["vehicle_fleet"].columns, ["CO2_Emissions_Kg_per_KM", "CO2_Emissions_g_km"])
        keys["distance_km"] = find_column_name(data_frames["routes_distance"].columns, ["Distance_KM", "Distance_km"])
        keys["fuel_consumption"] = find_column_name(data_frames["routes_distance"].columns, ["Fuel_Consumption_L", "Fuel_Consumption"])
        keys["tolls"] = find_column_name(data_frames["routes_distance"].columns, ["Toll_Charges_INR", "Toll_Charges"])
        keys["traffic_delay_hrs"] = find_column_name(data_frames["routes_distance"].columns, ["Traffic_Delay_Minutes", "Traffic_Delays_hours"])
    except KeyError as e:
        st.error(f"❌ Error: Missing CSV file: {e}"); return None, None, None, None

    key_order_id_delivery = find_column_name(data_frames["delivery_performance"].columns, ["Order_ID", "order_id"])
    df_merged = pd.merge(data_frames["orders"], data_frames["delivery_performance"], left_on=keys["order_id"], right_on=key_order_id_delivery, how="left")
    pending_orders_df = df_merged[df_merged[keys["delivery_status"]].isna()].copy()
    np.random.seed(42); pending_orders_df["Demand"] = np.random.randint(1, 4, size=len(pending_orders_df))

    if keys["vehicle_type"]:
        capacity_map = {"Van": 15, "Truck": 50, "Refrigerated Unit": 40, "Express Bike": 2}
        data_frames["vehicle_fleet"]["Capacity"] = data_frames["vehicle_fleet"][keys["vehicle_type"]].map(capacity_map).fillna(10)
    else: data_frames["vehicle_fleet"]["Capacity"] = 20

    return (pending_orders_df, data_frames["vehicle_fleet"], data_frames["routes_distance"], keys)

@st.cache_data
def get_mock_geodata(locations: list):
    """ Creates mock Lat/Lon data """
    geo_data = {"Mumbai": (19.0760, 72.8777), "Delhi": (28.7041, 77.1025), "Bangalore": (12.9716, 77.5946),
                "Chennai": (13.0827, 80.2707), "Kolkata": (22.5726, 88.3639), "Pune": (18.5204, 73.8567),
                "Hyderabad": (17.3850, 78.4867), "Ahmedabad": (23.0225, 72.5714)}
    np.random.seed(42)
    for loc in locations:
        if loc not in geo_data: geo_data[loc] = (np.random.uniform(12.0, 29.0), np.random.uniform(72.0, 89.0))
    return geo_data

@st.cache_data
def create_cost_matrix(_locations: list, geo_data: dict, routes_df, vehicle_co2_kg_km):
    """ Creates simulated cost, time, and CO2 matrices with exaggerated differences. """
    locations = list(_locations); num_locations = len(locations)
    MOCK_FUEL_PRICE_INR = 95.0
    avg_dist = routes_df[KEY_DIST].mean(); avg_dist = 1.0 if avg_dist == 0 else avg_dist
    avg_fuel_cost = routes_df[KEY_FUEL].mean() * MOCK_FUEL_PRICE_INR
    avg_toll_cost = routes_df[KEY_TOLLS].mean()
    avg_cost_per_km = (avg_fuel_cost + avg_toll_cost) / avg_dist
    avg_delay_hours = routes_df[KEY_TRAFFIC_DELAY].mean() / 60.0
    avg_speed_kph = avg_dist / (avg_delay_hours + 8)
    if avg_speed_kph == 0 or pd.isna(avg_speed_kph): avg_speed_kph = 40.0

    cost_matrix = np.zeros((num_locations, num_locations), dtype=int)
    time_matrix = np.zeros((num_locations, num_locations), dtype=int)
    co2_matrix = np.zeros((num_locations, num_locations), dtype=int)

    for i in range(num_locations):
        for j in range(num_locations):
            if i == j: continue
            loc1 = geo_data[locations[i]]; loc2 = geo_data[locations[j]]
            dist = int(np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2) * 111)
            base_time_mins = int((dist / avg_speed_kph) * 60)
            cost_matrix[i, j] = int(dist * avg_cost_per_km) + int(base_time_mins * 0.5)
            time_matrix[i, j] = base_time_mins + int((dist / 10)**1.5)
            co2_matrix[i, j] = int(dist * vehicle_co2_kg_km) + int(base_time_mins * 0.05 * vehicle_co2_kg_km)

    return {"cost": cost_matrix, "time": time_matrix, "co2": co2_matrix}


# --- 4. Optimization Logic (UPDATED RETURN VALUE) ---
def solve_vrp(all_matrices, demands, capacity, optimization_choice):
    """
    Solves the VRP using Google OR-Tools, explicitly setting the
    cost evaluator based on the optimization_choice.

    Returns:
        tuple: (Optimized route indices, Objective value of the solution) or (None, None).
    """
    data = {'demands': demands, 'vehicle_capacities': [capacity], 'num_vehicles': 1, 'depot': 0}
    num_locations = len(all_matrices['cost'])

    try:
        manager = pywrapcp.RoutingIndexManager(num_locations, data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def create_transit_callback(matrix):
            def transit_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index); to_node = manager.IndexToNode(to_index)
                return matrix[from_node][to_node]
            return transit_callback
        cost_callback_index = routing.RegisterTransitCallback(create_transit_callback(all_matrices['cost']))
        time_callback_index = routing.RegisterTransitCallback(create_transit_callback(all_matrices['time']))
        co2_callback_index = routing.RegisterTransitCallback(create_transit_callback(all_matrices['co2']))

        if optimization_choice == "Cost": routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index); print("Solver prioritizing COST")
        elif optimization_choice == "Time": routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index); print("Solver prioritizing TIME")
        else: routing.SetArcCostEvaluatorOfAllVehicles(co2_callback_index); print("Solver prioritizing CO2")

        def demand_callback(from_index): from_node = manager.IndexToNode(from_index); return data['demands'][from_node]
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimension(demand_callback_index, 0, data['vehicle_capacities'][0], True, 'Capacity')

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            route = []; index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index)); index = solution.Value(routing.NextVar(index))
            route.append(0)
            objective_value = solution.ObjectiveValue() # Get solver's objective
            return route, objective_value # Return both
        else: return None, None
    except Exception as e:
        st.error(f"❌ Optimization solver failed: {e}"); return None, None

# --- 5. Download Helper ---
def get_csv_download_link(df, filename="optimized-route.csv", link_text="⬇️ Download Route Plan"):
    """ Generates a CSV download link. """
    csv = df.to_csv(index=False); b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'


# --- 6. Main Application UI (Simple Layout) ---
def main():
    """ Renders the main Streamlit application. """
    st.title("🚚 NexGen RoutePrime: Smart Route Planner")
    st.markdown("Optimize routes for cost, time, and environmental impact.")

    # --- Initialize Session State (Add objective_value) ---
    if 'results_calculated' not in st.session_state:
        st.session_state.results_calculated = False; st.session_state.route_plan_df = None
        st.session_state.total_cost = 0; st.session_state.total_time = 0; st.session_state.total_co2 = 0
        st.session_state.objective_value = 0 # Added
        st.session_state.route_indices = None; st.session_state.all_locations = None
        st.session_state.geo_data = None; st.session_state.loc_to_orders = None
        st.session_state.depot_loc = None; st.session_state.opt_choice = "Cost"

    pending_df, fleet_df, routes_df, keys = load_all_data()

    if pending_df is None:
        st.warning("⚠️ Data loading failed. Check 'data' folder and CSVs."); st.stop()
    else:
        # --- Define Global Keys (Keep as before) ---
        global KEY_ORIGIN, KEY_DEST, KEY_VEHICLE_ID, KEY_VEHICLE_TYPE, KEY_CO2_KG_KM, KEY_DIST, KEY_FUEL, KEY_TOLLS, KEY_TRAFFIC_DELAY
        KEY_ORIGIN = keys.get("origin"); KEY_DEST = keys.get("destination")
        KEY_VEHICLE_ID = keys.get("vehicle_id"); KEY_VEHICLE_TYPE = keys.get("vehicle_type")
        KEY_CO2_KG_KM = keys.get("co2_kg_km"); KEY_DIST = keys.get("distance_km")
        KEY_FUEL = keys.get("fuel_consumption"); KEY_TOLLS = keys.get("tolls")
        KEY_TRAFFIC_DELAY = keys.get("traffic_delay_hrs")

        critical_keys_list = [
            (KEY_ORIGIN, "Origin"), (KEY_DEST, "Destination"), (KEY_VEHICLE_ID, "Vehicle_ID"),
            (KEY_VEHICLE_TYPE, "Vehicle_Type"), (KEY_CO2_KG_KM, "CO2_Emissions_Kg_per_KM"),
            (KEY_DIST, "Distance_KM"), (KEY_FUEL, "Fuel_Consumption_L"),
            (KEY_TOLLS, "Toll_Charges_INR"), (KEY_TRAFFIC_DELAY, "Traffic_Delay_Minutes")
        ]
        missing_keys = [name for key, name in critical_keys_list if key is None]
        if missing_keys:
            st.error("❌ Critical columns missing:"); st.json(missing_keys); st.stop()

        # --- Sidebar ---
        st.sidebar.header("Route Optimizer Controls")
        warehouse_list = [""] + sorted(list(pending_df[KEY_ORIGIN].unique()))
        selected_warehouse = st.sidebar.selectbox("1. Select Origin Warehouse", options=warehouse_list, index=0)

        vehicle_list = [""] + sorted(list(fleet_df[KEY_VEHICLE_ID].unique()))
        selected_vehicle_id = st.sidebar.selectbox(
            "2. Select Vehicle", options=vehicle_list, index=0,
            format_func=lambda x: f"{x} ({fleet_df[fleet_df[KEY_VEHICLE_ID] == x][KEY_VEHICLE_TYPE].values[0]})" if x else "Select..."
        )

        selected_orders = []
        if selected_warehouse:
            available_orders_df = pending_df[pending_df[KEY_ORIGIN] == selected_warehouse]
            if not available_orders_df.empty:
                selected_orders = st.sidebar.multiselect(
                    f"3. Select Pending Orders ({len(available_orders_df)} available)",
                    options=list(available_orders_df[keys["order_id"]]),
                )
            else: st.sidebar.warning(f"⚠️ No pending orders for {selected_warehouse}.")
        else: st.sidebar.info("ℹ️ Select warehouse first.")

        opt_choice = st.sidebar.selectbox("4. Optimize for:", options=["Cost", "Time", "Environmental Impact"], index=0)
        generate_button = st.sidebar.button("🚀 Generate Optimized Route", type="primary", use_container_width=True)

        # --- Main Page Layout ---
        col1, col2 = st.columns([2, 1])

        # --- Main Logic on Button Click ---
        if generate_button:
            if not selected_warehouse or not selected_vehicle_id or not selected_orders:
                st.error("⚠️ Please select Warehouse, Vehicle, and at least one Order.")
            else:
                with st.spinner("⏳ Processing inputs..."):
                    # ... (Data preparation logic remains the same) ...
                    vehicle = fleet_df[fleet_df[KEY_VEHICLE_ID] == selected_vehicle_id].iloc[0]
                    vehicle_capacity = int(vehicle["Capacity"])
                    vehicle_co2_kg_km = float(vehicle[KEY_CO2_KG_KM])
                    orders_to_route = available_orders_df[available_orders_df[keys["order_id"]].isin(selected_orders)]
                    total_demand = orders_to_route["Demand"].sum()
                    if total_demand > vehicle_capacity: st.error(f"❌ Demand ({total_demand}) exceeds capacity ({vehicle_capacity})."); st.stop()
                    depot_loc = selected_warehouse; customer_locs = list(orders_to_route[KEY_DEST].unique())
                    all_locations = [depot_loc] + customer_locs
                    loc_to_orders = {loc: list(orders_to_route[orders_to_route[KEY_DEST] == loc][keys["order_id"]]) for loc in customer_locs}
                    geo_data = get_mock_geodata(all_locations)
                    all_matrices = create_cost_matrix(all_locations, geo_data, routes_df, vehicle_co2_kg_km)
                    demands = [0] + [orders_to_route[orders_to_route[KEY_DEST] == loc]["Demand"].sum() for loc in customer_locs]

                with st.spinner(f"✨ Optimizing route for **{opt_choice}**..."):
                    # --- CALL UPDATED SOLVER (Capture objective_value) ---
                    route_indices, objective_value = solve_vrp(all_matrices, demands, vehicle_capacity, opt_choice)

                # --- Check if solver succeeded ---
                if route_indices is None: # Changed check from 'not route_indices'
                    st.error("❌ Optimization failed. No feasible route found."); st.stop()

                # --- Store Results (Include objective_value) ---
                st.session_state.results_calculated = True; st.session_state.route_indices = route_indices
                st.session_state.all_locations = all_locations; st.session_state.geo_data = geo_data
                st.session_state.loc_to_orders = loc_to_orders; st.session_state.depot_loc = depot_loc
                st.session_state.opt_choice = opt_choice
                st.session_state.objective_value = objective_value # Store objective

                # Calculate other totals based on the found route
                total_cost = 0; total_time = 0; total_co2 = 0
                for i in range(len(route_indices) - 1):
                    start = route_indices[i]; end = route_indices[i+1]
                    total_cost += all_matrices["cost"][start, end]
                    total_time += all_matrices["time"][start, end]
                    total_co2 += all_matrices["co2"][start, end]
                st.session_state.total_cost = total_cost; st.session_state.total_time = total_time; st.session_state.total_co2 = total_co2

                # Create route plan dataframe
                route_plan = []
                for i in range(len(route_indices)):
                    node = route_indices[i]; loc_name = all_locations[node]
                    orders = "DEPART/RETURN WAREHOUSE" if i == 0 or i == len(route_indices)-1 else loc_to_orders.get(loc_name, "N/A")
                    demand = demands[node] if i < len(route_indices)-1 else 0
                    route_plan.append({"Stop": i, "Location": loc_name, "Orders": orders, "Demand at Stop": demand })
                st.session_state.route_plan_df = pd.DataFrame(route_plan)

                st.success(f"✅ Route optimized successfully for **{opt_choice}**!")
                st.rerun()

        # --- Display Results ---
        if st.session_state.results_calculated:
            with col1:
                 st.subheader("Optimized Route Map")
                 route_coords = [st.session_state.geo_data[st.session_state.all_locations[i]] for i in st.session_state.route_indices]
                 m_route = folium.Map(location=st.session_state.geo_data[st.session_state.depot_loc], zoom_start=8, tiles='OpenStreetMap')
                 folium.Marker(location=st.session_state.geo_data[st.session_state.depot_loc], popup=f"Warehouse (Start/End)\n{st.session_state.depot_loc}", tooltip="Depot", icon=folium.Icon(color="blue", icon="industry", prefix="fa")).add_to(m_route)
                 for i in st.session_state.route_indices[1:-1]:
                     loc = st.session_state.all_locations[i]
                     folium.Marker(location=st.session_state.geo_data[loc], popup=f"Stop: {loc}\nOrders: {st.session_state.loc_to_orders.get(loc, 'N/A')}", tooltip=f"Stop {i}: {loc}", icon=folium.Icon(color="red", icon="box", prefix="fa")).add_to(m_route)
                 folium.PolyLine(route_coords, color="#FF4B4B", weight=5, opacity=0.9, tooltip="Optimized Path").add_to(m_route)
                 st_folium(m_route, width=None, height=500, key="optimized_map_results")

            with col2:
                 st.subheader(f"Route KPIs (Optimized for **{st.session_state.opt_choice}**)")
                 kpi_col1, kpi_col2 = st.columns(2)

                 # --- UPDATED KPI DISPLAY ---
                 # Display the actual optimized value prominently
                 if st.session_state.opt_choice == "Cost":
                     kpi_col1.metric("💰 **Optimized Cost (Sim.)**", f"₹{st.session_state.objective_value:,.0f}")
                     kpi_col2.metric("⏱️ Resulting Time (Sim.)", f"{st.session_state.total_time:,.0f} mins")
                     kpi_col1.metric("☁️ Resulting CO2 (Sim.)", f"{st.session_state.total_co2:,.1f} kg")
                 elif st.session_state.opt_choice == "Time":
                     kpi_col2.metric("⏱️ **Optimized Time (Sim.)**", f"{st.session_state.objective_value:,.0f} mins")
                     kpi_col1.metric("💰 Resulting Cost (Sim.)", f"₹{st.session_state.total_cost:,.0f}")
                     kpi_col1.metric("☁️ Resulting CO2 (Sim.)", f"{st.session_state.total_co2:,.1f} kg")
                 else: # Environmental Impact
                     kpi_col1.metric("☁️ **Optimized CO2 (Sim.)**", f"{st.session_state.objective_value:,.1f} kg")
                     kpi_col1.metric("💰 Resulting Cost (Sim.)", f"₹{st.session_state.total_cost:,.0f}")
                     kpi_col2.metric("⏱️ Resulting Time (Sim.)", f"{st.session_state.total_time:,.0f} mins")

                 kpi_col2.metric("📍 Total Stops", f"{len(st.session_state.route_indices) - 2}")
                 st.markdown("---")
                 st.dataframe(st.session_state.route_plan_df, height=350) # Slightly shorter table
                 st.markdown("---")
                 st.markdown(
                     get_csv_download_link(st.session_state.route_plan_df, f'route_{selected_warehouse}_{st.session_state.opt_choice}.csv'),
                     unsafe_allow_html=True
                 )
        else:
             # Placeholders
             with col1:
                 st.subheader("Route Map Preview")
                 map_center = get_mock_geodata(["Mumbai"])["Mumbai"]
                 m = folium.Map(location=map_center, zoom_start=8, tiles='OpenStreetMap')
                 st_folium(m, width=None, height=500, key="default_map_placeholder")
             with col2:
                 st.subheader("Route Plan & KPIs")
                 st.info("⬅️ Configure route in sidebar and click 'Generate Optimized Route'.")

# --- 7. App Entry Point ---
if __name__ == "__main__":
    main()