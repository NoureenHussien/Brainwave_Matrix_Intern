import streamlit as st
import plotly.express as px
import pandas as pd
import warnings
import datetime as dt

warnings.filterwarnings('ignore')

# --- Page Configuration and Global Styling ---
st.set_page_config(
    page_title="Superstore Sales Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define a professional color palette
COLOR_PALETTE = {
    "primary_blue": "#2874A6",
    "light_blue": "#AED6F1",
    "success_green": "#2ECC71",
    "warning_orange": "#F39C12",
    "danger_red": "#E74C3C",
    "background_light": "#F8F9FA",
    "card_background": "#FFFFFF",
    "text_dark": "#34495E",
    "text_light": "#7F8C8D",
    "border_light": "#E0E0E0",
    "sidebar_background": "#FFFFFF",
    "metric_border": "#3498DB"
}

# Apply custom CSS for a modern, clean, and consistent dashboard look
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLOR_PALETTE["background_light"]};
        font-family: 'Inter', sans-serif;
        color: {COLOR_PALETTE["text_dark"]};
    }}

    .st-emotion-cache-18ni7ap {{
        background-color: {COLOR_PALETTE["card_background"]};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 1rem 1rem;
    }}

    .st-emotion-cache-s2s16p {{
        background-color: {COLOR_PALETTE["light_blue"]}33;
        border-radius: 0.5rem;
        border: 1px solid {COLOR_PALETTE["light_blue"]};
        transition: all 0.2s ease-in-out;
    }}
    .st-emotion-cache-s2s16p:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }}

    .st-emotion-cache-1uqo6q {{
        background-color: {COLOR_PALETTE["card_background"]};
        border-radius: 0.75rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid {COLOR_PALETTE["border_light"]};
        transition: all 0.3s ease-in-out;
    }}
    .st-emotion-cache-1uqo6q:hover {{
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        transform: translateY(-5px);
    }}

    [data-testid="stSidebar"] {{
        background-color: {COLOR_PALETTE["sidebar_background"]};
        box-shadow: 2px 0 8px rgba(0,0,0,0.05);
        padding: 1.5rem;
    }}

    .st-emotion-cache-l9rw5y {{
        background-color: {COLOR_PALETTE["primary_blue"]};
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out, box-shadow 0.2s ease-in-out;
    }}
    .st-emotion-cache-l9rw5y:hover {{
        background-color: #1F618D;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}

    .st-emotion-cache-z5fcl4 {{
        background-color: {COLOR_PALETTE["card_background"]};
        border-bottom: 1px solid {COLOR_PALETTE["border_light"]};
        padding: 0.5rem 1rem;
        border-radius: 0.75rem 0.75rem 0 0;
    }}
    .st-emotion-cache-z5fcl4 button {{
        color: {COLOR_PALETTE["text_light"]};
        font-weight: 600;
        padding: 0.75rem 1.25rem;
        border-radius: 0.5rem;
        transition: all 0.2s ease-in-out;
    }}
    .st-emotion-cache-z5fcl4 button:hover {{
        background-color: {COLOR_PALETTE["background_light"]};
        color: {COLOR_PALETTE["primary_blue"]};
    }}
    .st-emotion-cache-z5fcl4 button[aria-selected="true"] {{
        background-color: {COLOR_PALETTE["primary_blue"]};
        color: white;
        border-bottom: none;
    }}

    [data-testid="stMetric"] {{
        background-color: {COLOR_PALETTE["card_background"]};
        border-left: 5px solid {COLOR_PALETTE["metric_border"]};
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease-in-out;
    }}
    [data-testid="stMetric"]:hover {{
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.9rem;
        color: {COLOR_PALETTE["text_light"]};
        font-weight: 600;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {COLOR_PALETTE["primary_blue"]};
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 0.9rem;
        font-weight: 600;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        background-color: {COLOR_PALETTE["background_light"]};
        border: 2px dashed {COLOR_PALETTE["border_light"]};
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s ease-in-out;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {COLOR_PALETTE["primary_blue"]};
        background-color: {COLOR_PALETTE["background_light"]}99;
    }}

    ::-webkit-scrollbar {{
        width: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLOR_PALETTE["background_light"]};
        border-radius: 10px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLOR_PALETTE["text_light"]};
        border-radius: 10px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLOR_PALETTE["text_dark"]};
    }}

    .stSlider > div > div > div[data-baseweb="slider"] {{
        background-color: {COLOR_PALETTE["border_light"]};
    }}
    .stSlider > div > div > div > div[data-baseweb="slider"] > div {{
        background-color: {COLOR_PALETTE["primary_blue"]};
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {COLOR_PALETTE["text_dark"]};
        font-weight: 600;
    }}

    .main-title {{
        font-size: 3.5rem;
        color: {COLOR_PALETTE["primary_blue"]};
        text-align: center;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
    }}

    </style>
""", unsafe_allow_html=True)

# --- Main Dashboard Title ---
st.markdown('<h class="main-title"> Superstore Sales Analytics Dashboard </h>', unsafe_allow_html=True)


# --- Data Loading and Initial Processing ---
if 'df' not in st.session_state:
    st.session_state.df = None

@st.cache_data(ttl=3600)
def load_and_process_data(file_content, file_extension):
    df = pd.DataFrame()

    try:
        if file_extension == 'csv' or file_extension == 'txt':
            df = pd.read_csv(file_content, encoding='latin1')
        elif file_extension == 'xlsx':
            df = pd.read_excel(file_content)
        else:
            st.error("Unsupported file type. Please upload a CSV, TXT, or XLSX file.")
            return pd.DataFrame()
        st.success(f"Successfully loaded data.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

    # --- Data Cleaning and Feature Engineering ---
    for col in ['Sales', 'Profit', 'Quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            st.warning(f"Column '{col}' not found in the dataset.")

    for col in ['Order Date', 'Ship Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            df.dropna(subset=[col], inplace=True)
        else:
            st.warning(f"Column '{col}' not found in the dataset.")

    df.drop_duplicates(inplace=True)

    if 'Postal Code' in df.columns:
        df['Postal Code'] = df['Postal Code'].astype(str)

    if 'Order Date' in df.columns and 'Ship Date' in df.columns:
        df['Shipping Duration'] = (df['Ship Date'] - df['Order Date']).dt.days
        df = df[df['Shipping Duration'] >= 0]
    else:
        st.warning("Could not calculate 'Shipping Duration'. 'Order Date' or 'Ship Date' missing.")


    if 'Discount' in df.columns:
        df = df[(df['Discount'] >= 0) & (df['Discount'] <= 1)]

    for col in ['Sales', 'Profit', 'Quantity', 'Shipping Duration']:
        if col in df.columns:
            df = df[pd.to_numeric(df[col], errors='coerce').notnull()]

    return df

# --- Sidebar File Uploader ---
st.sidebar.title("Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("Upload sales data", type=["csv", "txt", "xlsx"],
                                        help="Upload your sales data file in CSV, TXT, or XLSX format")

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    st.session_state.df = load_and_process_data(uploaded_file, file_extension)
else:
    if st.session_state.df is None:
        st.info("Please upload a sales data file to get started.")
        st.stop()

df = st.session_state.df

if df.empty:
    st.error("No valid data loaded. Please upload a correct file or check file content.")
    st.stop()

# --- Date Range Filter with Improved UI ---
min_order_date = df["Order Date"].min()
max_order_date = df["Order Date"].max()

st.sidebar.subheader("Date Range Filter")
selected_start_date = st.sidebar.date_input("Start date", min_order_date,
                                          min_value=min_order_date,
                                          max_value=max_order_date)
selected_end_date = st.sidebar.date_input("End date", max_order_date,
                                        min_value=min_order_date,
                                        max_value=max_order_date)

filtered_df = df[(df["Order Date"] >= pd.to_datetime(selected_start_date)) &
                 (df["Order Date"] <= pd.to_datetime(selected_end_date))]

# --- Geographic Filters with Icons ---
st.sidebar.subheader("Geographic Filters")
if 'Region' in filtered_df.columns:
    selected_region = st.sidebar.multiselect(
        "Select regions",
        options=filtered_df['Region'].unique().tolist(),
        key='region_filter'
    )
    if selected_region:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_region)]

if 'City' in filtered_df.columns:
    selected_city = st.sidebar.multiselect(
        "Select cities",
        options=filtered_df['City'].unique().tolist(),
        key='city_filter'
    )
    if selected_city:
        filtered_df = filtered_df[filtered_df['City'].isin(selected_city)]

if filtered_df.empty:
    st.warning("No data found for the selected filters. Please adjust your selections.")
    st.stop()

# --- Define Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Product & Sales Analysis",
    "Customer Segmentation",
    "Shipping Insights",
    "Advanced Data Explorer"
])

# --- Tab 1: Product & Sales Analysis ---
with tab1:
    st.markdown("## Sales and Profit Overview")

    # Key Performance Indicators (KPIs)
    total_sales = filtered_df['Sales'].sum()
    total_profit = filtered_df['Profit'].sum()
    profit_ratio = (total_profit / total_sales) * 100 if total_sales else 0
    total_orders = filtered_df['Order ID'].nunique()

    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Sales", value=f"${total_sales:,.2f}")
    with col2:
        st.metric(label="Total Profit", value=f"${total_profit:,.2f}")
    with col3:
        st.metric(label="Profit Ratio", value=f"{profit_ratio:,.2f}%")
    with col4:
        st.metric(label="Total Orders", value=f"{total_orders:,.0f}")

    st.markdown("### Monthly Sales Trend")
    if 'Order Date' in filtered_df.columns and 'Sales' in filtered_df.columns:
        monthly_sales = filtered_df.set_index('Order Date').resample('M')['Sales'].sum().reset_index()
        monthly_sales['Month'] = monthly_sales['Order Date'].dt.to_period('M').astype(str)
        fig_monthly_sales = px.line(monthly_sales, x='Month', y='Sales',
                                    markers=True,
                                    line_shape="linear",
                                    render_mode="svg",
                                    color_discrete_sequence=[COLOR_PALETTE["primary_blue"]])
        fig_monthly_sales.update_layout(xaxis_title="Month", yaxis_title="Total Sales")
        st.plotly_chart(fig_monthly_sales, use_container_width=True)
    st.markdown("### Sales Distribution by Category and Region")
    col5, col6 = st.columns(2)
    with col5:
        if 'Category' in filtered_df.columns and 'Sub-Category' in filtered_df.columns:
            fig_category = px.sunburst(
                filtered_df,
                path=['Category', 'Sub-Category'],
                values='Sales',
                title='Sales Breakdown by Product Category and Sub-Category',
                color_continuous_scale=[COLOR_PALETTE["light_blue"], COLOR_PALETTE["primary_blue"]])
            st.plotly_chart(fig_category, use_container_width=True)
        elif 'Category' in filtered_df.columns:
            sales_by_category = filtered_df.groupby('Category')['Sales'].sum().reset_index()
            fig_category = px.sunburst(sales_by_category,
                                        path=['Category'],
                                        values='Sales',
                                        title='Sales by Product Category',
                                        color_continuous_scale=[COLOR_PALETTE["light_blue"], COLOR_PALETTE["primary_blue"]])
            st.plotly_chart(fig_category, use_container_width=True)
        else:
            st.warning("Category or Sub-Category column not found for sunburst chart.")

    with col6:
        if 'Region' in filtered_df.columns:
            sales_by_region = filtered_df.groupby('Region')['Sales'].sum().reset_index()
            fig_region = px.pie(sales_by_region, values='Sales', names='Region',
                                title='Sales Distribution by Region',
                                hole=0.3,
                                color_discrete_sequence=[COLOR_PALETTE["primary_blue"], COLOR_PALETTE["light_blue"], "#77AADD", "#5E8BB6"])
            st.plotly_chart(fig_region, use_container_width=True)
        else:
            st.warning("Region column not found.")

    st.markdown("### Product Performance and Profitability")

    if 'Sales' in filtered_df.columns and 'Profit' in filtered_df.columns and 'Quantity' in filtered_df.columns and 'Category' in filtered_df.columns:
        fig_sales_profit_scatter = px.scatter(filtered_df, x='Sales', y='Profit', size='Quantity', color='Category',
                                            hover_name='Product Name',
                                            labels={'Sales': 'Total Sales', 'Profit': 'Total Profit'},
                                            height=500,
                                            color_discrete_sequence=px.colors.qualitative.Plotly)
        st.plotly_chart(fig_sales_profit_scatter, use_container_width=True)
    else:
        st.warning("Required columns (Sales, Profit, Quantity, Category) not found for scatter plot.")


    st.markdown("### Top cities by Sales and Profit")
    if 'City' in filtered_df.columns:
        city_data = filtered_df.groupby('City').agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).nlargest(30, 'Sales').reset_index()

        fig = px.bar(city_data,
                    x='City',
                    y=['Sales', 'Profit'],
                    barmode='group',
                    labels={'value': 'Amount ($)', 'variable': 'Metric', 'City': 'City'},
                    color_discrete_map={'Sales': COLOR_PALETTE["primary_blue"], 'Profit': COLOR_PALETTE["success_green"]})

        fig.update_layout(
            xaxis_tickangle=-90,
            yaxis=dict(tickprefix='$'),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            hoverlabel=dict(bgcolor='white')
        )
        st.plotly_chart(fig, use_container_width=True)
# --- Tab 2: Customer Segmentation ---
with tab2:
    st.markdown("## Customer Segmentation Analysis")
    required_cols = {'Customer Name', 'Order Date', 'Sales'}
    if required_cols.issubset(filtered_df.columns):

        # Step 1: RFM Calculation
        snapshot_date = filtered_df['Order Date'].max() + dt.timedelta(days=1)
        rfm = filtered_df.groupby('Customer Name').agg(
            Recency=('Order Date', lambda date: (snapshot_date - date.max()).days),
            Frequency=('Order ID', 'nunique'),
            Monetary=('Sales', 'sum')
        ).reset_index()

        # Step 2: RFM Scoring using qcut with fallbacks
        def assign_scores(column, ascending=True):
            n_unique = column.nunique()
            q_val = min(5, n_unique)
            if q_val < 2:
                return pd.Series([1] * len(column), index=column.index)
            try:
                labels = list(range(q_val, 0, -1)) if ascending else list(range(1, q_val + 1))
                return pd.qcut(column, q=q_val, labels=labels, duplicates='drop').astype(int)
            except ValueError:
                # fallback: assign all to 1 if qcut fails
                return pd.Series([1] * len(column), index=column.index)

        rfm['R_Score'] = assign_scores(rfm['Recency'], ascending=True)
        rfm['F_Score'] = assign_scores(rfm['Frequency'], ascending=False)
        rfm['M_Score'] = assign_scores(rfm['Monetary'], ascending=False)

        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

        # Step 3: RFM Segmentation
        def rfm_segment(row):
            if row['R_Score'] == 5 and row['F_Score'] == 5 and row['M_Score'] == 5:
                return "Champions"
            elif row['R_Score'] >= 4 and row['F_Score'] >= 4:
                return "Loyal Customers"
            elif row['R_Score'] >= 4 and row['F_Score'] >= 2 and row['M_Score'] >= 2:
                return "Potential Loyalists"
            elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
                return "Promising"
            elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
                return "At-Risk / Lost"
            elif row['R_Score'] <= 2 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
                return "Need Attention"
            else:
                return "Others"

        rfm['Segment'] = rfm.apply(rfm_segment, axis=1)
        # Step 4: Segment Pie Chart
        st.markdown("#### Customer Segments Distribution")
        segment_counts = rfm['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig_segment = px.pie(segment_counts, values='Count', names='Segment',
                             hole=0.3, color_discrete_sequence=px.colors.sequential.Blues[::-1])
        st.plotly_chart(fig_segment, use_container_width=True)

        # Step 5: RFM Score Distribution Charts
        st.markdown("#### Stacked Distribution of RFM Scores by Customer Segments")

        rfm_melt = rfm.melt(id_vars=['Segment'], value_vars=['R_Score', 'F_Score', 'M_Score'],
                    var_name='Score_Type', value_name='Score')

        fig_stacked = px.histogram(rfm_melt, x='Segment', color='Score', barmode='stack',
                                   facet_col='Score_Type', category_orders={"Score": [1, 2, 3, 4, 5]},
                                   color_discrete_sequence=px.colors.sequential.Viridis)

        st.plotly_chart(fig_stacked, use_container_width=True)


        # Step 6: Explanation
        with st.expander("Explanation of Customer Segments"):
            st.write("""
            - **Champions:** Recently bought, buy often, and spend the most.
            - **Loyal Customers:** Frequently buy and spend well.
            - **Potential Loyalists:** Recent buyers with moderate frequency and value.
            - **Promising:** Moderate on all RFM scores.
            - **At-Risk / Lost:** Long time no purchase, low frequency, low spending.
            - **Need Attention:** Good spenders and frequent, but not recent.
            - **Others:** Do not fall clearly in other categories.
            """)

# --- Tab 3: Geographic Performance ---
with tab3:

    st.markdown("## Shipping Insights")
    st.markdown("### Shipping Performance Analysis Overview")
    if 'Shipping Duration' in filtered_df.columns:

        avg_shipping_duration = filtered_df['Shipping Duration'].mean()
        total_shipments = len(filtered_df)
        profit_per_shipment = filtered_df['Profit'].sum() / total_shipments if total_shipments else 0

        on_time_deliveries = filtered_df[filtered_df['Shipping Duration'] <= 7].shape[0]
        on_time_rate = (on_time_deliveries / total_shipments) * 100 if total_shipments else 0

        col_ship1, col_ship2, col_ship3, col_ship4 = st.columns(4)

        with col_ship1:
            st.metric(label="Avg. Shipping Duration (Days)", value=f"{avg_shipping_duration:,.1f}")
        with col_ship2:
            st.metric(label="Total Shipments", value=f"{total_shipments:,.0f}")
        with col_ship3:
            st.metric(label="Profit Per Shipment", value=f"${profit_per_shipment:,.2f}")
        with col_ship4:
            st.metric(label="On-Time Delivery Rate (<=7 days)", value=f"{on_time_rate:,.1f}%")
        st.markdown("### Distribution of Shipping Duration")
        fig_shipping_dist = px.histogram(
                filtered_df,
                x='Shipping Duration',
                marginal="box",
                color='Shipping Duration',
                color_discrete_sequence=px.colors.sequential.Blues[::-1]
            )
        fig_shipping_dist.update_layout(
                xaxis_title="Shipping Duration (Days)",
                yaxis_title="Number of Shipments"
            )
        st.plotly_chart(fig_shipping_dist, use_container_width=True)

        ship_data = filtered_df.copy()
        if 'Region' in ship_data.columns and 'Ship Mode' in ship_data.columns and 'Order ID' in ship_data.columns and 'Shipping Duration' in ship_data.columns:
            ship_data = ship_data.groupby(['Region', 'Ship Mode']).agg({
                'Order ID': 'count',
                'Shipping Duration': 'mean'
            }).reset_index()
            fig = px.treemap(ship_data,
                        path=['Region', 'Ship Mode'],
                        values='Order ID',
                        color='Shipping Duration',
                        color_continuous_scale=[COLOR_PALETTE["light_blue"], COLOR_PALETTE["primary_blue"]],
                        title='<b>Shipping Performance by Region & Ship Mode</b><br>Size = Number of Orders | Color = Avg. Shipping Days',
                        hover_data={'Shipping Duration':':.1f days'},
                        custom_data=['Region', 'Ship Mode'])

            fig.update_traces(
                textinfo="label+value",
                texttemplate='<b>%{label}</b><br>%{value} orders',
                hovertemplate=(
                    '<b>%{customdata[1]}</b> (%{customdata[0]})<br>'
                    'Orders: %{value}<br>'
                    'Avg. Shipping: %{color:.1f} days<extra></extra>'
                )
            )

            fig.update_layout(
                coloraxis_colorbar=dict(
                    title='Days',
                    thickness=20,
                    len=0.75,
                    yanchor='middle',
                    y=0.5
                ),
                margin=dict(t=80, l=0, r=0, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Advanced Data Explorer ---
with tab4:
    st.markdown("## Data Exploration and Raw Data View")

    st.subheader("Monthly Sales Performance by Sub-Category")
    with st.expander("View Sales Performance by Sub-Category Across Months"):
        filtered_df["month"] = filtered_df["Order Date"].dt.month_name()
        monthly_sub_category_sales = pd.pivot_table(
            data=filtered_df,
            values="Sales",
            index=["Sub-Category"],
            columns="month",
            aggfunc='sum'
        ).fillna(0)

        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        monthly_sub_category_sales = monthly_sub_category_sales.reindex(columns=month_order, fill_value=0)

        fig_heatmap = px.imshow(
            monthly_sub_category_sales,
            text_auto=".2s",
            aspect="auto",
            color_continuous_scale=[COLOR_PALETTE["background_light"], COLOR_PALETTE["primary_blue"]],
        )
        fig_heatmap.update_xaxes(side="top")
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.dataframe(monthly_sub_category_sales.style.format('${:,.2f}'), use_container_width=True)
        csv_heatmap = monthly_sub_category_sales.to_csv().encode('utf-8')
        st.download_button(
            'Download Monthly Sub-Category Sales Heatmap Data',
            data=csv_heatmap,
            file_name="monthly_sub_category_sales.csv",
            mime="text/csv",
            key='download_heatmap_data'
        )

    st.markdown("### Raw Data Viewer ")

    if 'Customer Name' in filtered_df.columns and 'Customer Name' in rfm.columns:
        filtered_df_with_rfm = pd.merge(filtered_df, rfm[['Customer Name', 'Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'Segment']],
                                        on='Customer Name',
                                        how='left')
    else:
        st.warning("Customer Name column not found for merging RFM data.")
        filtered_df_with_rfm = filtered_df.copy()

    pd.set_option("styler.render.max_elements", max(350000, filtered_df_with_rfm.size))

    st.dataframe(filtered_df_with_rfm.head(300).style.format({
        "Sales": '${:,.2f}',
        "Profit": '${:,.2f}',
        "Quantity": '{:,.0f}',
        "Order Date": '{:%Y-%m-%d}',
        "Recency": '{:,.0f} days',
        "Frequency": '{:,.0f}',
        "Monetary": '${:,.2f}'
    }), use_container_width=True)

    csv_filtered_with_rfm = filtered_df_with_rfm.to_csv(index=False).encode('utf-8')
    st.download_button(
        'Download Filtered Data ',
        data=csv_filtered_with_rfm,
        file_name="filtered_superstore_data.csv",
        mime="text/csv",
        help='Download the currently filtered dataset as a CSV file'
    )

    csv_original = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        'Download Full Original Dataset',
        data=csv_original,
        file_name="full_original_superstore_data.csv",
        mime="text/csv",
        help='Download the entire original dataset (before any date/geographic filters)'
    )
