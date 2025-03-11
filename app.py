import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st
import base64
import matplotlib.pyplot as plt

# Function to set background image
def set_background(local_image_path):
    with open(local_image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/jpeg;base64,{encoded}") no-repeat center center fixed;
                background-size: cover;
                opacity: 0.9;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Set the background using a local image
set_background("pic.jpg")

# Load datasets
@st.cache_data
def load_data():
    final_colour_df = pd.read_csv("Final_Colour.csv")
    machine_capacity_df = pd.read_excel("Machine_Capacity.xlsx")
    return final_colour_df, machine_capacity_df

final_colour_df, machine_capacity_df = load_data()

# Clean column names
final_colour_df.columns = final_colour_df.columns.str.strip()
machine_capacity_df.columns = machine_capacity_df.columns.str.strip()

# Rename columns for consistency
machine_capacity_df.rename(columns={'PIP Category': 'intPIPCategoryID', 'MC No': 'MCNo', 'Color Shade': 'RecipeColorCategory'}, inplace=True)

# Assign PIP Category Names
pip_category_mapping = {
    1: "Scallop Elastic",
    2: "Fold Over Elastic",
    3: "Plush Elastic",
    4: "Jacquard Elastic",
    5: "Bra Tape",
    6: "Polyester Scallop Elastic"
}
machine_capacity_df['PIP Category Name'] = machine_capacity_df['intPIPCategoryID'].map(pip_category_mapping)

# Handle missing vcColourNo values
final_colour_df["vcColourNo"] = final_colour_df.groupby(["MCNo", "RecipeColorCategory", "intPIPCategoryID"])['vcColourNo']\
    .transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# Compute RFT Score
rft_scores = final_colour_df.groupby(['MCNo', 'RecipeColorCategory', 'intPIPCategoryID', 'vcColourNo'])\
    .agg(RFT_Score=('RFT', lambda x: (x == 1).sum() / len(x)))\
    .reset_index()

# Merge RFT scores with machine capacity dataset
recommendation_df = machine_capacity_df.merge(rft_scores, on=['MCNo', 'RecipeColorCategory', 'intPIPCategoryID'], how='left')

def assign_rft_score(machine_no, category, shade):
    similar_machines = recommendation_df[
        (recommendation_df['intPIPCategoryID'] == category) & 
        (recommendation_df['RecipeColorCategory'] == shade) & 
        (recommendation_df['RFT_Score'].notna())
    ]
    if not similar_machines.empty:
        return similar_machines['RFT_Score'].mean()
    return 0.5  # Default RFT Score for new machines

recommendation_df.loc[recommendation_df['RFT_Score'].isna(), 'RFT_Score'] = recommendation_df.apply(
    lambda row: assign_rft_score(row['MCNo'], row['intPIPCategoryID'], row['RecipeColorCategory']), axis=1
)

# Compute Success Score using the validated formula
recommendation_df['Success_Score'] = recommendation_df.apply(lambda row: (row['RFT_Score'] * 100) / (1 + abs(row['Capacity'] - 10000)), axis=1)

# Train Gradient Boosting Model for Capacity & RFT Weighting
X = recommendation_df[['Capacity', 'RFT_Score']]
Y = recommendation_df['Success_Score']
model = GradientBoostingRegressor()
model.fit(X, Y)
capacity_weight, rft_weight = model.feature_importances_

def compute_machine_score(capacity, rft_score, quantity):
    capacity_diff = abs(capacity - quantity)
    score = (capacity_weight * (1 / (1 + capacity_diff))) + (rft_weight * rft_score)
    return score

recommendation_df['Raw_Machine_Score'] = recommendation_df.apply(lambda row: compute_machine_score(row['Capacity'], row['RFT_Score'], 10000), axis=1)

# Normalize scores to 100%
scaler = MinMaxScaler(feature_range=(0, 100))
recommendation_df['Machine_Score'] = scaler.fit_transform(recommendation_df[['Raw_Machine_Score']])

# Machine Recommendation Function
def recommend_machines(width, color_shade, pip_category, color_no, quantity, top_n=5):
    filtered_machines = recommendation_df[
        (recommendation_df['Width'] == width) &
        (recommendation_df['RecipeColorCategory'] == color_shade) &
        (recommendation_df['intPIPCategoryID'] == pip_category)
    ]
    
    if color_no and color_no in filtered_machines['vcColourNo'].values:
        filtered_machines = filtered_machines[filtered_machines['vcColourNo'] == color_no]
    
    new_machines = recommendation_df[recommendation_df['MCNo'].isin(['N16-006D', 'N17-006E'])]
    if not new_machines.empty:
        filtered_machines = pd.concat([filtered_machines, new_machines]).drop_duplicates()
    
    unique_machines = filtered_machines.drop_duplicates(subset=['MCNo']).sort_values(by='Machine_Score', ascending=False)
    
    return unique_machines.head(top_n)[['MCNo', 'No Of Tapes', 'Speed', 'RFT_Score', 'Capacity', 'Machine_Score']]

# Streamlit Dashboard
def main():
    # Centered Title
    st.markdown("""
        <h1 style="text-align:center; font-size:50px; color:#000000;">
        Tape Dye Machine Recommendation System
        </h1>
        """, unsafe_allow_html=True)

    st.sidebar.title("Filters & Preferences")
    
    width = st.sidebar.selectbox("Select Width Category", recommendation_df['Width'].unique())
    color_shade = st.sidebar.selectbox("Select Color Shade", recommendation_df['RecipeColorCategory'].unique())
    pip_category_name = st.sidebar.selectbox("Select PIP Category", machine_capacity_df['PIP Category Name'].unique())
    pip_category = machine_capacity_df[machine_capacity_df['PIP Category Name'] == pip_category_name]['intPIPCategoryID'].values[0]
    
    color_no_options = recommendation_df['vcColourNo'].dropna().unique()
    color_no = st.sidebar.selectbox("Select Color Number", ['Unknown'] + list(color_no_options))
    color_no = None if color_no == 'Unknown' else color_no
    
    reference_color_no = st.sidebar.selectbox("Select a Reference Color No (optional, if new color)", ['None'] + list(color_no_options))
    reference_color_no = None if reference_color_no == 'None' else reference_color_no
    
    quantity = st.sidebar.number_input("Enter Quantity (m)", min_value=1, value=10000)
    top_n = st.sidebar.slider("Select Number of Machines to Recommend", min_value=1, max_value=19, value=5)
    
    if st.sidebar.button("Get Recommendations"):
        if color_no is None and reference_color_no:
            st.warning("Color Number is unknown! Using Reference Color No for recommendation.")
            color_no = reference_color_no

        recommendations = recommend_machines(width, color_shade, pip_category, color_no, quantity, top_n)

        # Centered Table (Recommendations)
        st.markdown("""
            <style>
            .dataframe { 
                margin: auto;
                text-align: center;
            }
            </style>
            """, unsafe_allow_html=True)

        st.subheader("Recommended Machines", divider='rainbow')
        st.dataframe(recommendations)

        # Generate a bar chart (Machine Scores)
        if "MCNo" in recommendations.columns and "Machine_Score" in recommendations.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(recommendations["MCNo"], recommendations["Machine_Score"], color="#F5DEB3")
            ax.set_xlabel("Machine Score")
            ax.set_ylabel("Machine No")
            ax.set_title("Top Recommended Machines by Machine Score")
            ax.invert_yaxis()

            # Center the chart
            st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Columns 'MCNo' or 'Machine_Score' not found in DataFrame!")

if __name__ == "__main__":
    main()
