import pandas as pd
import streamlit as st
import plotly.express as px
from io import BytesIO

# Configuration
st.set_page_config(layout="wide")

# Constants Configuration
CONFIG = {
    'revolving_suppliers': [
        "PFIZER OVERSEAS LLC", "SANOFI WINTHROP INDUSTRIE", "SERUM INSTITUTE INC",
        "MSD LATIN AMERICA SERVICES S.DE R.L.", "GLAXO SMITHKLINE BIOLOGICALS SA",
        "MERCK SHARP & DOHME LLC", "GC BIOPHARMA CORP.", "BIOLOGICAL E LIMITED.",
        "ABU DHABI MEDICAL DEVICES CO."
    ],
    'strategic_suppliers': [
        "VALENT BIOSCIENCES CORPORATION", "Gilead Sciences Ireland UC", "CEPHEID HBDC SAS",
        "HETERO LABS LIMITED", "MACLEODS PHARMACEUTICALS LTD.", "ABBOTT RAPID DX INTERNATIONAL LIMITED",
        "MYLAN LABORATORIES LTD", "SANOFI WINTHROP INDUSTRIE (SWIND)",
        "SHENZHEN MINDRAY BIO-MEDICAL ELECTRONICS CO., LTD", "LUPIN LIMITED", "AMEX",
        "Missionpharma A/S", "REMEDICA LTD", "Unimed Procurement Services Ltd"
    ],
    'small_country_iso': ['ATG', 'DMA', 'GRD', 'KNA', 'LCA', 'VCT'],
    'country_iso_mapping': {
        # North America
        'Mexico': 'MEX',
        
        # Central America
        'Belize': 'BLZ', 
        'Costa Rica': 'CRI',
        'El Salvador': 'SLV',
        'Guatemala': 'GTM',
        'Honduras': 'HND',
        'Nicaragua': 'NIC',
        'Panama': 'PAN',
        
        # Caribbean
        'Anguilla': 'AIA',
        'Antigua and Barbuda': 'ATG',
        'Aruba': 'ABW',
        'Bahamas': 'BHS',
        'Barbados': 'BRB',
        'Bermuda': 'BMU',
        'British Virgin Islands': 'VGB',
        'Cayman Islands': 'CYM',
        'Cuba': 'CUB',
        'Curaçao': 'CUW',
        'Dominica': 'DMA',
        'Dominican Republic': 'DOM',
        'Grenada': 'GRD',
        'Haiti': 'HTI',
        'Jamaica': 'JAM',
        'Saint Kitts and Nevis': 'KNA',
        'Saint Lucia': 'LCA',
        'Saint Vincent and the Grenadines': 'VCT',
        'Sint Maarten': 'SXM',
        'Trinidad and Tobago': 'TTO',
        'Turks and Caicos Islands': 'TCA',
        
        # South America
        'Argentina': 'ARG',
        'Bolivia': 'BOL',
        'Brazil': 'BRA',
        'Chile': 'CHL',
        'Colombia': 'COL',
        'Ecuador': 'ECU',
        'Guyana': 'GUY',
        'Paraguay': 'PRY',
        'Peru': 'PER',
        'Suriname': 'SUR',
        'Uruguay': 'URY',
        'Venezuela': 'VEN',
        
        # Territories
        'Puerto Rico': 'PRI',
        'French Guiana': 'GUF',
        'Guadeloupe': 'GLP',
        'Martinique': 'MTQ'
    }
}

# Define supplier lists from CONFIG
REVOLVING_SUPPLIERS = CONFIG['revolving_suppliers']
STRATEGIC_SUPPLIERS = CONFIG['strategic_suppliers']

@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess the PAHO data"""
    cols = ['YearReceipt', 'Supplier', 'ShippingMethod', 'ShipToAddressCountry',
            'PurchaseOrderNumber', 'Freight per APO', 'Amount per PO Line',
            'LineDescription', 'FundType']
    
    df = pd.read_excel(uploaded_file, sheet_name='Data2024', usecols=cols)
    
    # Handle duplicates
    duplicate_mask = df.duplicated(['PurchaseOrderNumber', 'Freight per APO'], keep='first')
    df.loc[duplicate_mask, 'Freight per APO'] = None
    
    return df

def validate_data(df):
    """Validate the input data structure and content"""
    required_columns = ['YearReceipt', 'Supplier', 'ShippingMethod', 
                       'ShipToAddressCountry', 'PurchaseOrderNumber', 
                       'Freight per APO', 'Amount per PO Line', 
                       'LineDescription', 'FundType']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    if df['YearReceipt'].isnull().any():
        st.warning("Data contains rows with missing year information")
    
    return True

def calculate_shipping_percentages(df):
    """Calculate air vs ocean shipping percentages"""
    df['ShippingMethod'] = df['ShippingMethod'].str.upper().str.strip()
    shipping_df = df[df['ShippingMethod'].isin(['AIR', 'OCEAN'])]
    
    method_breakdown = (
        shipping_df.groupby(['Supplier', 'ShippingMethod'])['Amount per PO Line']
        .sum()
        .unstack(fill_value=0)
        .rename(columns={'AIR': 'Air', 'OCEAN': 'Ocean'})
    )
    
    supplier_totals = method_breakdown.sum(axis=1)
    method_breakdown['Air %'] = (method_breakdown['Air'] / supplier_totals * 100).round(1)
    method_breakdown['Ocean %'] = (method_breakdown['Ocean'] / supplier_totals * 100).round(1)
    
    return method_breakdown[['Air %', 'Ocean %']].reset_index()

def process_country_data(fund_df):
    """Process country-level data with freight ratios"""
    country_df = (
        fund_df.groupby(['Supplier', 'ShipToAddressCountry'])
        .agg({'Amount per PO Line': 'sum', 'Freight per APO': 'sum'})
        .reset_index()
    )
    country_df['Freight Ratio (%)'] = (
        country_df['Freight per APO'] / country_df['Amount per PO Line'] * 100
    ).round(1)
    return country_df

def build_final_report(df):
    """Generate the complete report DataFrame"""
    # Filter data by fund type
    rev_fund = df[(df['YearReceipt'] == 2024) & (df['FundType'] == 'Revolving Fund')]
    strat_fund = df[(df['YearReceipt'] == 2024) & (df['FundType'] == 'Strategic Fund')]
    
    # Filter by specified suppliers
    rev_filtered = rev_fund[rev_fund['Supplier'].isin(REVOLVING_SUPPLIERS)]
    strat_filtered = strat_fund[strat_fund['Supplier'].isin(STRATEGIC_SUPPLIERS)]
    
    # Calculate shipping percentages
    rev_shipping = calculate_shipping_percentages(rev_fund)
    strat_shipping = calculate_shipping_percentages(strat_fund)
    
    # Process country-level data
    rev_country = process_country_data(rev_fund)
    strat_country = process_country_data(strat_fund)
    
    # Build the final report
    report_data = []
    
    def add_supplier_rows(suppliers, fund_df, shipping_df, country_df):
        for supplier in suppliers:
            if supplier not in fund_df['Supplier'].unique():
                continue
                
            # Get products
            products = fund_df[fund_df['Supplier'] == supplier]['LineDescription'].unique()
            product_list = '\n'.join([f"• {p}" for p in products])
            
            # Get shipping percentages
            shipping = shipping_df[shipping_df['Supplier'] == supplier]
            air_pct = f"{shipping['Air %'].values[0]:.1f}%" if not shipping.empty else "0%"
            ocean_pct = f"{shipping['Ocean %'].values[0]:.1f}%" if not shipping.empty else "0%"
            
            # Get country data
            countries = country_df[country_df['Supplier'] == supplier]
            country_lines = []
            freight_lines = []
            ratio_lines = []
            ratio_values = []
            
            for _, row in countries.iterrows():
                country = row['ShipToAddressCountry']
                amount = row['Amount per PO Line']
                freight = row['Freight per APO']
                ratio = row['Freight Ratio (%)']
                
                country_lines.append(f"{country}: ${amount:,.2f}")
                freight_lines.append(f"{country}: ${freight:,.2f}")
                ratio_lines.append(f"{country}: {ratio:.1f}%")
                ratio_values.append(ratio)
            
            report_data.append({
                'Supplier': supplier,
                'Products Delivered 2024': product_list,
                'Air shipment, %': air_pct,
                'Sea shipment, %': ocean_pct,
                'PAHO Destination Countries and Annual Spend': '\n'.join(country_lines),
                'Total Freight per PAHO Country': '\n'.join(freight_lines),
                'Freight-to-Product Ratio per PAHO Country': '\n'.join(ratio_lines),
                '_Ratio_Values': ratio_values,
                '_FundType': 'Revolving Fund' if supplier in REVOLVING_SUPPLIERS else 'Strategic Fund',
                '_Country_Data': countries[['ShipToAddressCountry', 'Amount per PO Line', 'Freight per APO', 'Freight Ratio (%)']]
            })
    
    add_supplier_rows(REVOLVING_SUPPLIERS, rev_filtered, rev_shipping, rev_country)
    add_supplier_rows(STRATEGIC_SUPPLIERS, strat_filtered, strat_shipping, strat_country)
    
    return pd.DataFrame(report_data)

def generate_excel(final_report):
    """Generate Excel file with conditional formatting"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Export without internal columns
        final_report.drop(columns=['_Ratio_Values', '_FundType', '_Country_Data']).to_excel(
            writer, index=False, sheet_name='Supplier Report'
        )
        
        workbook = writer.book
        worksheet = writer.sheets['Supplier Report']
        
        # Define formats
        wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 
            'bg_color': '#D9E1F2', 'border': 1
        })
        red_bold_format = workbook.add_format({
            'bold': True, 'font_color': 'red',
            'text_wrap': True, 'valign': 'top'
        })
        
        # Apply conditional formatting to ratio column
        ratio_col = 6  # Column G
        for row_idx in range(1, len(final_report)+1):
            ratio_text = final_report.at[row_idx-1, 'Freight-to-Product Ratio per PAHO Country']
            ratio_values = final_report.at[row_idx-1, '_Ratio_Values']
            country_ratios = ratio_text.split('\n')
            
            rich_parts = []
            for i, (ratio, value) in enumerate(zip(country_ratios, ratio_values)):
                if i > 0:
                    rich_parts.append('\n')
                
                country, ratio_val = ratio.split(': ')
                if value >= 50:
                    rich_parts.extend([red_bold_format, f"{country}: ", red_bold_format, ratio_val])
                else:
                    rich_parts.append(ratio)
            
            worksheet.write_rich_string(row_idx, ratio_col, *rich_parts, wrap_format)
        
        # Set column widths
        worksheet.set_column('A:A', 30, wrap_format)
        worksheet.set_column('B:B', 40, wrap_format)
        worksheet.set_column('C:D', 15, wrap_format)
        worksheet.set_column('E:E', 35, wrap_format)
        worksheet.set_column('F:G', 30, wrap_format)
        
        # Format headers
        for col_num, value in enumerate(final_report.drop(columns=['_Ratio_Values', '_FundType', '_Country_Data']).columns):
            worksheet.write(0, col_num, value, header_format)
        
        worksheet.freeze_panes(1, 0)
    
    return output.getvalue()

def create_supplier_dashboard(supplier_data):
    """Enhanced dashboard with small country visibility solutions"""
    supplier_name = supplier_data['Supplier'].iloc[0]
    st.subheader(f"📊 {supplier_name} Performance Dashboard")
    
    # Prepare country data
    country_data = supplier_data['_Country_Data'].iloc[0].copy()
    country_data['Ratio Category'] = country_data['Freight Ratio (%)'].apply(
        lambda x: 'High (≥50%)' if x >= 50 else 'Normal (<50%)'
    )
    country_data['ISO'] = country_data['ShipToAddressCountry'].map(CONFIG['country_iso_mapping'])
    
    # Identify small countries with high ratios
    small_high_ratio = country_data[
        (country_data['ISO'].isin(CONFIG['small_country_iso'])) &
        (country_data['Ratio Category'] == 'High (≥50%)')
    ]
    
    # Create dashboard tabs
    tab1, tab2, tab3 = st.tabs(["🌍 Country Analysis", "📦 Freight Insights", "✈️ Shipping Methods"])
    
    with tab1:
        # Country Analysis
        st.markdown("### Product Spend by Country")
        fig1 = px.bar(country_data.sort_values('Amount per PO Line', ascending=False),
                     x='ShipToAddressCountry',
                     y='Amount per PO Line',
                     color='Amount per PO Line',
                     color_continuous_scale='teal',
                     labels={'ShipToAddressCountry': 'Country', 'Amount per PO Line': 'Total Spend ($)'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with tab2:
        st.markdown("### Freight Ratio by Country")
        
        # Create two-column layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Enhanced map showing all Latin America
            fig2 = px.choropleth(
                country_data.dropna(subset=['ISO']),
                locations="ISO",
                color="Ratio Category",
                color_discrete_map={
                    'High (≥50%)': 'orange',
                    'Normal (<50%)': 'blue'
                },
                hover_name="ShipToAddressCountry",
                hover_data={
                    'Amount per PO Line': ':$,.2f',
                    'Freight per APO': ':$,.2f',
                    'Freight Ratio (%)': ':.1f%',
                    'ISO': False
                },
                projection="natural earth",
                scope="south america",  # Show all Latin America
                height=600
            )
            
            # Add special markers for small high-ratio countries
            if not small_high_ratio.empty:
                fig2.add_scattergeo(
                    locations=small_high_ratio['ISO'],
                    text=small_high_ratio['ShipToAddressCountry'],
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    name='Small Country Alert',
                    hoverinfo='text+name',
                    mode='markers+text',
                    textposition='top center'
                )
            
            # Adjust map view to include all of Latin America
            fig2.update_geos(
                showcountries=True,
                countrycolor="lightgray",
                showocean=True,
                oceancolor="lightblue",
                lataxis_range=[-60, 35],  # From southern Chile to northern Mexico
                lonaxis_range=[-120, -30]  # From Pacific to Atlantic
            )
            
            fig2.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # High-ratio alerts panel - SIMPLIFIED VERSION
            st.markdown("**🚨 High Freight Ratio Alerts**")
            
            # Get all high ratio countries
            high_ratio = country_data[country_data['Ratio Category'] == 'High (≥50%)']
            
            if not high_ratio.empty:
                # Sort by ratio descending
                high_ratio = high_ratio.sort_values('Freight Ratio (%)', ascending=False)
                
                for _, row in high_ratio.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div style="
                            background-color: #fff3f3;
                            padding: 10px;
                            border-radius: 5px;
                            margin-bottom: 10px;
                            border-left: 4px solid #ff4b4b;
                        ">
                            <b>{row['ShipToAddressCountry']}</b><br>
                            Freight Ratio: <b>{row['Freight Ratio (%)']:.1f}%</b><br>
                            Spend: ${row['Amount per PO Line']:,.2f}<br>
                            Freight: ${row['Freight per APO']:,.2f}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("No high ratio countries found")
            
            # Unmapped countries warning
            unmapped = country_data[country_data['ISO'].isna()]
            if not unmapped.empty:
                st.warning(f"Missing map data for: {', '.join(unmapped['ShipToAddressCountry'].tolist())}")
    
    with tab3:
        # Shipping Methods
        st.markdown("### Shipping Method Distribution")
        try:
            air_pct = float(supplier_data['Air shipment, %'].iloc[0].replace('%', ''))
            ocean_pct = float(supplier_data['Sea shipment, %'].iloc[0].replace('%', ''))
            
            fig3 = px.pie(
                names=['Air', 'Ocean'],
                values=[air_pct, ocean_pct],
                color=['Air', 'Ocean'],
                color_discrete_map={'Air': '#5E72E4', 'Ocean': '#2DCE89'},
                hole=0.4
            )
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(fig3, use_container_width=True)
            with col2:
                st.metric("Air Shipping", f"{air_pct:.1f}%")
                st.metric("Ocean Shipping", f"{ocean_pct:.1f}%")
                
        except (ValueError, AttributeError):
            st.warning("Shipping method data not available")
    
    # Products section
    st.markdown("---")
    st.markdown("### 🛒 Products Delivered")
    st.markdown(supplier_data['Products Delivered 2024'].iloc[0])

def show_help():
    """Interactive help system"""
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        ## PAHO Spend Analysis Tool Guide
        
        **Purpose**: Analyze supplier spending patterns with focus on freight costs.
        
        **Features**:
        - Supplier performance dashboards
        - Freight ratio analysis
        - Shipping method breakdowns
        - Country-level spending insights
        
        **How To Use**:
        1. Upload your PAHO spend data file
        2. Select a fund type (Revolving/Strategic)
        3. Choose a supplier to analyze
        4. Explore the interactive dashboard
        5. Download the full report
        """)

def main():
    st.title("PAHO Spend Analysis Dashboard")
    show_help()
    
    uploaded_file = st.file_uploader(
        "Upload 'PAHO Spend Data 22-24.xlsx'", 
        type="xlsx"
    )
    
    if uploaded_file:
        with st.spinner("Processing data..."):
            try:
                df = load_data(uploaded_file)
                validate_data(df)
                final_report = build_final_report(df)
                excel_data = generate_excel(final_report)
                
                st.success("Analysis complete!")
                
                # Download button
                st.download_button(
                    label="Download Full Report",
                    data=excel_data,
                    file_name="PAHO_Supplier_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Step 1: Select fund type
                fund_type = st.radio(
                    "Select Fund Type:",
                    ["Revolving Fund", "Strategic Fund"],
                    horizontal=True
                )
                
                # Filter suppliers based on fund type
                if fund_type == "Revolving Fund":
                    suppliers = final_report[final_report['_FundType'] == 'Revolving Fund']['Supplier'].unique()
                else:
                    suppliers = final_report[final_report['_FundType'] == 'Strategic Fund']['Supplier'].unique()
                
                # Step 2: Select supplier
                selected_supplier = st.selectbox(
                    f"Select a {fund_type} Supplier:",
                    suppliers
                )
                
                # Show dashboard for selected supplier
                if selected_supplier:
                    create_supplier_dashboard(final_report[final_report['Supplier'] == selected_supplier])
                
                # Show full data tables in expanders
                with st.expander(f"Show All {fund_type} Suppliers Data"):
                    fund_df = final_report[final_report['_FundType'] == fund_type].drop(
                        columns=['_Ratio_Values', '_FundType', '_Country_Data'])
                    st.dataframe(fund_df, height=(len(fund_df) * 35 + 38))
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()