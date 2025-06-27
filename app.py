import pandas as pd
import streamlit as st
import plotly.express as px
from io import BytesIO
from typing import Dict, List

# Configuration
st.set_page_config(layout="wide", page_title="PAHO Spend Analysis", page_icon="🌎")

# Constants Configuration
CONFIG = {
    'default_revolving_suppliers': [
        "PFIZER OVERSEAS LLC", "SANOFI WINTHROP INDUSTRIE", "SERUM INSTITUTE INC",
        "MSD LATIN AMERICA SERVICES S.DE R.L.", "GLAXO SMITHKLINE BIOLOGICALS SA",
        "MERCK SHARP & DOHME LLC", "GC BIOPHARMA CORP.", "BIOLOGICAL E LIMITED.",
        "ABU DHABI MEDICAL DEVICES CO."
    ],
    'default_strategic_suppliers': [
        "VALENT BIOSCIENCES CORPORATION", "Gilead Sciences Ireland UC", "CEPHEID HBDC SAS",
        "HETERO LABS LIMITED", "MACLEODS PHARMACUTICALS LTD.", "ABBOTT RAPID DX INTERNATIONAL LIMITED",
        "MYLAN LABORATORIES LTD", "SANOFI WINTHROP INDUSTRIE (SWIND)",
        "SHENZHEN MINDRAY BIO-MEDICAL ELECTRONICS CO., LTD", "LUPIN LIMITED", "AMEX",
        "Missionpharma A/S", "REMEDICA LTD", "Unimed Procurement Services Ltd"
    ],
    'country_iso_mapping': {
        'Mexico': 'MEX', 'Belize': 'BLZ', 'Costa Rica': 'CRI', 'El Salvador': 'SLV',
        'Guatemala': 'GTM', 'Honduras': 'HND', 'Nicaragua': 'NIC', 'Panama': 'PAN',
        'Anguilla': 'AIA', 'Antigua and Barbuda': 'ATG', 'Aruba': 'ABW', 'Bahamas': 'BHS',
        'Barbados': 'BRB', 'Bermuda': 'BMU', 'British Virgin Islands': 'VGB', 'Cayman Islands': 'CYM',
        'Cuba': 'CUB', 'Curaçao': 'CUW', 'Dominica': 'DMA', 'Dominican Republic': 'DOM',
        'Grenada': 'GRD', 'Haiti': 'HTI', 'Jamaica': 'JAM', 'Saint Kitts and Nevis': 'KNA',
        'Saint Lucia': 'LCA', 'Saint Vincent and the Grenadines': 'VCT', 'Sint Maarten': 'SXM',
        'Trinidad and Tobago': 'TTO', 'Turks and Caicos Islands': 'TCA', 'Argentina': 'ARG',
        'Bolivia': 'BOL', 'Brazil': 'BRA', 'Chile': 'CHL', 'Colombia': 'COL', 'Ecuador': 'ECU',
        'Guyana': 'GUY', 'Paraguay': 'PRY', 'Peru': 'PER', 'Suriname': 'SUR', 'Uruguay': 'URY',
        'Venezuela': 'VEN', 'Puerto Rico': 'PRI', 'French Guiana': 'GUF', 'Guadeloupe': 'GLP',
        'Martinique': 'MTQ'
    }
}

@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess the PAHO data"""
    cols = ['YearReceipt', 'Supplier', 'ShippingMethod', 'ShipToAddressCountry',
            'PurchaseOrderNumber', 'Freight per APO', 'Amount per PO Line',
            'LineDescription', 'FundType']
    
    try:
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        df_list = []
        
        for sheet_name, sheet_df in sheets.items():
            if sheet_name.startswith('Data20'):
                temp_df = sheet_df[cols].copy()
                temp_df['DataYear'] = sheet_name.replace('Data', '')
                df_list.append(temp_df)
        
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            df['YearReceipt'] = df['YearReceipt'].fillna(df['DataYear'])
        else:
            df = pd.read_excel(uploaded_file, sheet_name='Data2024', usecols=cols)
            df['DataYear'] = '2024'
        
        df['ShippingMethod'] = df['ShippingMethod'].str.upper().str.strip()
        duplicate_mask = df.duplicated(['PurchaseOrderNumber', 'Freight per APO'], keep='first')
        df.loc[duplicate_mask, 'Freight per APO'] = None
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def validate_data(df):
    """Validate the required columns exist"""
    required_columns = ['YearReceipt', 'Supplier', 'ShippingMethod', 
                       'ShipToAddressCountry', 'PurchaseOrderNumber', 
                       'Freight per APO', 'Amount per PO Line', 
                       'LineDescription', 'FundType']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    return True

@st.cache_data
def calculate_shipping_percentages(df):
    """Calculate shipping method percentages"""
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
    """Process country-level data"""
    country_df = (
        fund_df.groupby(['Supplier', 'ShipToAddressCountry'])
        .agg({'Amount per PO Line': 'sum', 'Freight per APO': 'sum'})
        .reset_index()
    )
    country_df['Freight Ratio (%)'] = (country_df['Freight per APO'] / country_df['Amount per PO Line'] * 100).round(1)
    return country_df

def build_final_report(df, selected_years, revolving_suppliers, strategic_suppliers):
    """Build the final report DataFrame"""
    report_data = []
    
    for year in selected_years:
        year_data = df[df['YearReceipt'] == year]
        revolving_total = year_data[year_data['FundType'] == 'Revolving Fund']['Amount per PO Line'].sum()
        strategic_total = year_data[year_data['FundType'] == 'Strategic Fund']['Amount per PO Line'].sum()
        
        for fund_type in ['Revolving Fund', 'Strategic Fund']:
            fund_data = year_data[year_data['FundType'] == fund_type]
            current_fund_total = revolving_total if fund_type == 'Revolving Fund' else strategic_total
            
            vendors = fund_data['Supplier'].unique()
            if fund_type == 'Revolving Fund' and revolving_suppliers:
                vendors = [v for v in vendors if v in revolving_suppliers]
            elif fund_type == 'Strategic Fund' and strategic_suppliers:
                vendors = [v for v in vendors if v in strategic_suppliers]

            shipping_pct = calculate_shipping_percentages(fund_data)
            country_data = process_country_data(fund_data)
            
            for vendor in vendors:
                vendor_data = fund_data[fund_data['Supplier'] == vendor]
                if vendor_data.empty:
                    continue
                    
                supplier_total = vendor_data['Amount per PO Line'].sum()
                products = vendor_data['LineDescription'].unique()
                product_list = '\n'.join([f"• {p}" for p in products])
                
                vendor_shipping = shipping_pct[shipping_pct['Supplier'] == vendor]
                air_pct = f"{vendor_shipping['Air %'].values[0]:.1f}%" if not vendor_shipping.empty else "N/A"
                ocean_pct = f"{vendor_shipping['Ocean %'].values[0]:.1f}%" if not vendor_shipping.empty else "N/A"
                
                countries = country_data[country_data['Supplier'] == vendor]
                country_lines = []
                freight_lines = []
                ratio_lines = []
                ratio_values = []
                
                for _, row in countries.iterrows():
                    country_lines.append(f"{row['ShipToAddressCountry']}: ${row['Amount per PO Line']:,.2f}")
                    freight_lines.append(f"{row['ShipToAddressCountry']}: ${row['Freight per APO']:,.2f}")
                    ratio_lines.append(f"{row['ShipToAddressCountry']}: {row['Freight Ratio (%)']:.1f}%")
                    ratio_values.append(row['Freight Ratio (%)'])
                
                report_data.append({
                    'Year': year,
                    'Supplier': vendor,
                    'Products Delivered': product_list,
                    'Air shipment, %': air_pct,
                    'Sea shipment, %': ocean_pct,
                    'PAHO Destination Countries and Annual Spend': '\n'.join(country_lines),
                    'Total Freight per PAHO Country': '\n'.join(freight_lines),
                    'Freight-to-Product Ratio per PAHO Country': '\n'.join(ratio_lines),
                    '_Ratio_Values': ratio_values,
                    '_FundType': fund_type,
                    '_Country_Data': countries[['ShipToAddressCountry', 'Amount per PO Line', 'Freight per APO', 'Freight Ratio (%)']],
                    '_Total_Amount': supplier_total,
                    '_Total_Freight': countries['Freight per APO'].sum(),
                    '_Fund_Total': current_fund_total
                })
    
    return pd.DataFrame(report_data)

def generate_star_rating_formula(indicator_col: str, row_offset: int = 2) -> str:
    """Generate Excel formula for star ratings"""
    return (
        f'=IF(INDEX(Indicator!{indicator_col}:{indicator_col},MATCH($A{row_offset},Indicator!$A:$A,0))=0,"",'
        f'REPT("★",MIN(5,INT(INDEX(Indicator!{indicator_col}:{indicator_col},MATCH($A{row_offset},Indicator!$A:$A,0))))) & '
        f'IF(INDEX(Indicator!{indicator_col}:{indicator_col},MATCH($A{row_offset},Indicator!$A:$A,0))-'
        f'INT(INDEX(Indicator!{indicator_col}:{indicator_col},MATCH($A{row_offset},Indicator!$A:$A,0)))>0.5,"☆",""))'
    )

def add_traffic_light_formatting(worksheet, last_row, col_letter='K'):
    """Add traffic light icon set conditional formatting to specified column"""
    # Define the icon set format - green circle for 1, red circle for 0
    worksheet.conditional_format(
        f'{col_letter}2:{col_letter}{last_row + 1}',
        {
            'type': 'icon_set',
            'icon_style': '3_traffic_lights',
            'icons': [
                {'criteria': '>=', 'type': 'number', 'value': 1, 'icon': 'green_circle'},
                {'criteria': '<', 'type': 'number', 'value': 1, 'icon': 'red_circle'}
            ],
            'reverse_icons': False,
            'icons_only': True
        }
    )

def create_dashboard_sheet(writer: pd.ExcelWriter, df: pd.DataFrame, fund_type: str, selected_year: str):
    """Create a complete dashboard sheet with star ratings"""
    workbook = writer.book
    
    # Prepare supplier totals
    supplier_totals = df.groupby('Supplier').agg({
        '_Total_Amount': 'sum',
        '_Total_Freight': 'sum',
        '_Fund_Total': 'first'
    }).reset_index().sort_values('_Total_Amount', ascending=False)

    # Create DataFrame with all columns
    dashboard_df = pd.DataFrame({
        'Supplier': supplier_totals['Supplier'],
        f'Supplier {selected_year} Spend': supplier_totals['_Total_Amount'],
        f'% of Total {selected_year} {fund_type} Fund': (supplier_totals['_Total_Amount'] / supplier_totals['_Fund_Total']).round(4),
        'Supplier Criticality Rating': "",
        'Year of Latest Sustainability Report': "",
        'Supplier Spend Profile': supplier_totals.apply(
            lambda row: (
                f"Annual Freight Cost: ${row['_Total_Freight']:,.2f}\n"
                f"Freight-to-Product Ratio: {((row['_Total_Freight'] / row['_Total_Amount']) * 100 if row['_Total_Amount'] > 0 else 0):.1f}%"
            ),
            axis=1
        ),
        'Production Efficiency/ Supplier Sustainability Profile': "",
        '% Renewable Energy': "",
        'Logistics Efficiency Profile': "",
        'Packaging Optimization Profile': "",
        'Regional Distribution Capacity Profile': "",
        'Innovation Profile': "",
        'Follow up Action Steps/Collaboration Opportunities': ""
    })

    # Write to Excel first
    sheet_name = f'Dashboard_{fund_type}'
    dashboard_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=1)
    worksheet = writer.sheets[sheet_name]

    # Define all formatting
    header_format = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 'border': 1
    })
    
    # Column formats (F-M)
    col_formats = {
        'F': workbook.add_format({'bg_color': '#BFC8D0', 'bold': True, 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'G': workbook.add_format({'bg_color': '#1AC56C', 'bold': True, 'text_wrap': True, 'valign': 'top', 'border': 1, 'font_color': 'white'}),
        'H': workbook.add_format({'bg_color': '#1AC56C', 'bold': True, 'text_wrap': True, 'valign': 'top', 'border': 1, 'font_color': 'white'}),
        'I': workbook.add_format({'bg_color': '#059FDB', 'bold': True, 'text_wrap': True, 'valign': 'top', 'border': 1, 'font_color': 'white'}),
        'J': workbook.add_format({'bg_color': '#013D59', 'bold': True, 'text_wrap': True, 'valign': 'top', 'border': 1, 'font_color': 'white'}),
        'K': workbook.add_format({'bg_color': '#74ACE0', 'bold': True, 'text_wrap': True, 'valign': 'top', 'border': 1, 'font_color': 'white'}),
        'L': workbook.add_format({'bg_color': '#FFBA00', 'bold': True, 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'M': workbook.add_format({'bg_color': '#BFC8D0', 'bold': True, 'text_wrap': True, 'valign': 'top', 'border': 1})
    }
    
    # Data row formats
    even_row_formats = {
        'F': workbook.add_format({'bg_color': '#F4F4F4', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'G': workbook.add_format({'bg_color': '#ECFAF2', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'H': workbook.add_format({'bg_color': '#ECFAF2', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'I': workbook.add_format({'bg_color': '#EBF7FC', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'J': workbook.add_format({'bg_color': '#EAEEF0', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'K': workbook.add_format({'bg_color': '#F2F7FD', 'text_wrap': True, 'valign': 'top','align': 'center', 'border': 1}),
        'L': workbook.add_format({'bg_color': '#FFF9ED', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'M': workbook.add_format({'bg_color': '#F4F4F4', 'text_wrap': True, 'valign': 'top', 'border': 1})
    }
    
    odd_row_formats = {
        'F': workbook.add_format({'bg_color': '#EDEDED', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'G': workbook.add_format({'bg_color': '#D8F4E5', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'H': workbook.add_format({'bg_color': '#D8F4E5', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'I': workbook.add_format({'bg_color': '#D7EDF8', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'J': workbook.add_format({'bg_color': '#D4DCE0', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'K': workbook.add_format({'bg_color': '#E5EFFA', 'text_wrap': True, 'valign': 'top','align': 'center', 'border': 1}),
        'L': workbook.add_format({'bg_color': '#FFF2DA', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'M': workbook.add_format({'bg_color': '#EDEDED', 'text_wrap': True, 'valign': 'top', 'border': 1})
    }

    # Write headers with formatting
    headers = [
        'Supplier', f'Supplier {selected_year} Spend', f'% of Total {selected_year} {fund_type} Fund',
        'Supplier Criticality Rating', 'Year of Latest Sustainability Report', 'Supplier Spend Profile',
        'Production Efficiency/ Supplier Sustainability Profile', '% Renewable Energy',
        'Logistics Efficiency Profile', 'Packaging Optimization Profile',
        'Regional Distribution Capacity Profile', 'Innovation Profile',
        'Follow up Action Steps/Collaboration Opportunities'
    ]
    
    for col_num, header in enumerate(headers):
        if col_num >= 5:  # Columns F-M
            col_letter = chr(65 + col_num)
            worksheet.write(0, col_num, header, col_formats.get(col_letter, header_format))
        else:
            worksheet.write(0, col_num, header, header_format)

    # Write data with formatting and formulas
    for row_idx in range(1, len(dashboard_df) + 1):
        for col_num in range(len(headers)):
            col_letter = chr(65 + col_num)
            cell_value = dashboard_df.iloc[row_idx-1, col_num]
            
            # Apply special formulas for star rating columns
            if headers[col_num] == 'Production Efficiency/ Supplier Sustainability Profile':
                formula = generate_star_rating_formula('R', row_idx+1)
                worksheet.write_formula(row_idx, col_num, formula, 
                                      even_row_formats[col_letter] if row_idx % 2 == 0 else odd_row_formats[col_letter])
            elif headers[col_num] == 'Logistics Efficiency Profile':
                formula = generate_star_rating_formula('V', row_idx+1)
                worksheet.write_formula(row_idx, col_num, formula,
                                      even_row_formats[col_letter] if row_idx % 2 == 0 else odd_row_formats[col_letter])
            elif headers[col_num] == 'Packaging Optimization Profile':
                formula = generate_star_rating_formula('AA', row_idx+1)
                worksheet.write_formula(row_idx, col_num, formula,
                                      even_row_formats[col_letter] if row_idx % 2 == 0 else odd_row_formats[col_letter])
            elif headers[col_num] == 'Innovation Profile':
                formula = generate_star_rating_formula('AI', row_idx+1)
                worksheet.write_formula(row_idx, col_num, formula,
                                      even_row_formats[col_letter] if row_idx % 2 == 0 else odd_row_formats[col_letter])
            elif headers[col_num] == 'Regional Distribution Capacity Profile':
                formula = f'=IF(INDEX(Indicator!AD:AD,MATCH($A{row_idx+1},Indicator!$A:$A,0))="Yes",1,0)'
                worksheet.write_formula(row_idx, col_num, formula,
                                      even_row_formats[col_letter] if row_idx % 2 == 0 else odd_row_formats[col_letter])
            else:
                # Regular data cells
                if col_letter in ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']:
                    if row_idx % 2 == 0:
                        worksheet.write(row_idx, col_num, cell_value, even_row_formats.get(col_letter))
                    else:
                        worksheet.write(row_idx, col_num, cell_value, odd_row_formats.get(col_letter))
                else:
                    worksheet.write(row_idx, col_num, cell_value)

    # Set column widths
    column_widths = {
        'A': 30, 'B': 60, 'C': 25, 'D': 25, 'E': 25,
        'F': 40, 'G': 25, 'H': 20, 'I': 25, 'J': 25,
        'K': 25, 'L': 20, 'M': 30
    }
    
    for col_letter, width in column_widths.items():
        worksheet.set_column(f'{col_letter}:{col_letter}', width)
    
    # Apply number formats
    money_format = workbook.add_format({'num_format': '$#,##0'})
    percent_format = workbook.add_format({'num_format': '0.0%'})
    worksheet.set_column('B:B', None, money_format)
    worksheet.set_column('C:C', None, percent_format)
    
    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, 0, len(headers)-1)

    add_traffic_light_formatting(worksheet, len(dashboard_df), 'K')

def create_indicator_sheet(writer: pd.ExcelWriter, working_df: pd.DataFrame, ratio_threshold: int):
    """Create the Indicator sheet with complete formatting"""
    workbook = writer.book
    
    # Create DataFrame with all columns
    export_df = working_df[[
        'Supplier', 'Products Delivered', 'Air shipment, %', 'Sea shipment, %',
        'PAHO Destination Countries and Annual Spend', 'Total Freight per PAHO Country',
        'Freight-to-Product Ratio per PAHO Country'
    ]].copy()
    
    # Add all additional columns
    additional_columns = [
        'Average Price USD/Product', 'Lead Time',
        'Sustainability Report (0.5)', 'ISO 140001 (0.5)',
        'Participate in regional or interregional initiatives (1)',
        'Supplier\'s carbon calculation (1)', 'Emission reduction targets (1)',
        'Supplier Specific Emission Factor (Aga Khan Tool)',
        'Product Specific Emission Factor (1)', 'Renewable Electricity (%)',
        'Sustainability Score (5)', 'Sea shipments for the non temperature controlled products (2)',
        'Sea shipments for vaccines (2)', 'Exploring transportation low emission solutions (1)',
        'Logistics Score (5)', 'Use of sustainable, eco-friendly materials (2)',
        'Minimizing material use in packaging (1)', 'Digital Solutions for Packaging (1)',
        'Recycling programs (1)', 'Packaging Score (5)', 'Regional warehouse facilities',
        'Regional production facilities', 'Regional Capacity Score (Y/N)', 'SAF/SMF (1)',
        'Green Corridors (1)', 'Green IT (1)', 'Other Innovation (2)',
        'Innovation Score (5)', 'Follow up Action Steps / Collaboration Opportunities'
    ]
    
    for col in additional_columns:
        export_df[col] = "0"
    
    # Write to Excel
    export_df.to_excel(writer, sheet_name='Indicator', index=False)
    worksheet = writer.sheets['Indicator']
    
    # Define all formats
    formats = {
        'light_blue': workbook.add_format({'bg_color': '#A6C9EC', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'light_gray': workbook.add_format({'bg_color': '#BFC8D0', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'green': workbook.add_format({'bg_color': '#1AC56C', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'theme2': workbook.add_format({'bg_color': '#E8E8E8', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'blue': workbook.add_format({'bg_color': '#059FDB', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'dark_blue': workbook.add_format({'bg_color': '#013D59', 'text_wrap': True, 'valign': 'top', 'border': 1, 'font_color': 'white'}),
        'light_blue2': workbook.add_format({'bg_color': '#74ACE0', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'yellow': workbook.add_format({'bg_color': '#FFBA00', 'text_wrap': True, 'valign': 'top', 'border': 1}),
        'wrap': workbook.add_format({'text_wrap': True, 'valign': 'top', 'border': 1}),
        'red_bold': workbook.add_format({'bold': True, 'font_color': 'red', 'text_wrap': True, 'valign': 'top', 'border': 1})
    }
    
    # Header formats mapping
    header_formats = {
        'Supplier': formats['light_blue'],
        'Products Delivered': formats['light_gray'],
        'Air shipment, %': formats['light_gray'],
        'Sea shipment, %': formats['light_gray'],
        'PAHO Destination Countries and Annual Spend': formats['light_gray'],
        'Total Freight per PAHO Country': formats['light_gray'],
        'Freight-to-Product Ratio per PAHO Country': formats['light_gray'],
        'Average Price USD/Product': formats['light_gray'],
        'Lead Time': formats['light_gray'],
        'Sustainability Report (0.5)': formats['green'],
        'ISO 140001 (0.5)': formats['green'],
        'Participate in regional or interregional initiatives (1)': formats['green'],
        'Supplier\'s carbon calculation (1)': formats['green'],
        'Emission reduction targets (1)': formats['green'],
        'Supplier Specific Emission Factor (Aga Khan Tool)': formats['green'],
        'Product Specific Emission Factor (1)': formats['green'],
        'Renewable Electricity (%)': formats['green'],
        'Sustainability Score (5)': formats['theme2'],
        'Sea shipments for the non temperature controlled products (2)': formats['blue'],
        'Sea shipments for vaccines (2)': formats['blue'],
        'Exploring transportation low emission solutions (1)': formats['blue'],
        'Logistics Score (5)': formats['theme2'],
        'Use of sustainable, eco-friendly materials (2)': formats['dark_blue'],
        'Minimizing material use in packaging (1)': formats['dark_blue'],
        'Digital Solutions for Packaging (1)': formats['dark_blue'],
        'Recycling programs (1)': formats['dark_blue'],
        'Packaging Score (5)': formats['theme2'],
        'Regional warehouse facilities': formats['light_blue2'],
        'Regional production facilities': formats['light_blue2'],
        'Regional Capacity Score (Y/N)': formats['theme2'],
        'SAF/SMF (1)': formats['yellow'],
        'Green Corridors (1)': formats['yellow'],
        'Green IT (1)': formats['yellow'],
        'Other Innovation (2)': formats['yellow'],
        'Innovation Score (5)': formats['theme2'],
        'Follow up Action Steps / Collaboration Opportunities': formats['light_gray']
    }
    
    # Write headers with formats
    for col_num, header in enumerate(export_df.columns):
        worksheet.write(0, col_num, header, header_formats.get(header, formats['light_gray']))

    # Write data with formatting
    for row_idx in range(1, len(working_df)+1):
        # Format multi-line columns
        for col_name in ['PAHO Destination Countries and Annual Spend',
                       'Total Freight per PAHO Country',
                       'Freight-to-Product Ratio per PAHO Country',
                       'Products Delivered']:
            col_idx = export_df.columns.get_loc(col_name)
            cell_value = working_df.at[row_idx-1, col_name]
            worksheet.write(row_idx, col_idx, cell_value, formats['wrap'])
        
        # Apply threshold formatting to ratio column
        ratio_col_idx = export_df.columns.get_loc('Freight-to-Product Ratio per PAHO Country')
        ratio_text = working_df.at[row_idx-1, 'Freight-to-Product Ratio per PAHO Country']
        ratio_values = working_df.at[row_idx-1, '_Ratio_Values']
        
        rich_parts = []
        for i, (ratio, value) in enumerate(zip(ratio_text.split('\n'), ratio_values)):
            if i > 0:
                rich_parts.append('\n')
            
            country, ratio_val = ratio.split(': ')
            if value >= ratio_threshold:
                rich_parts.extend([formats['red_bold'], f"{country}: ", formats['red_bold'], ratio_val])
            else:
                rich_parts.append(ratio)
        
        worksheet.write_rich_string(row_idx, ratio_col_idx, *rich_parts, formats['wrap'])

    # Set column widths
    column_widths = {
        'A': 30, 'B': 40, 'C': 15, 'D': 15, 'E': 35,
        'F': 30, 'G': 30, 'H': 20, 'I': 15, 'AJ': 40
    }
    
    for col_letter, width in column_widths.items():
        worksheet.set_column(f'{col_letter}:{col_letter}', width)
    
    worksheet.freeze_panes(1, 0)

def generate_excel(final_report: pd.DataFrame, ratio_threshold: int = 50, selected_year: str = "2024") -> bytes:
    """Generate complete Excel report with dashboards and indicator sheet"""
    output = BytesIO()
    
    with pd.ExcelWriter(
        output,
        engine='xlsxwriter',
        engine_kwargs={'options': {'strings_to_formulas': False}}
    ) as writer:
        workbook = writer.book
        working_df = final_report.copy()

        # Create empty Indicator sheet first (will be populated later)
        pd.DataFrame().to_excel(writer, sheet_name='Indicator')
        
        # Create dashboards
        revolving_df = working_df[working_df['_FundType'] == 'Revolving Fund']
        if not revolving_df.empty:
            create_dashboard_sheet(writer, revolving_df, 'Revolving Fund', selected_year)
        
        strategic_df = working_df[working_df['_FundType'] == 'Strategic Fund']
        if not strategic_df.empty:
            create_dashboard_sheet(writer, strategic_df, 'Strategic Fund', selected_year)
        
        # Create Indicator sheet (now that dashboards are done)
        create_indicator_sheet(writer, working_df, ratio_threshold)

    return output.getvalue()

def create_supplier_dashboard(supplier_data: pd.DataFrame, ratio_threshold: int) -> None:
    """Create interactive supplier dashboard"""
    if supplier_data.empty:
        st.error("No supplier data available")
        return

    supplier_name = supplier_data['Supplier'].iloc[0]
    year = supplier_data['Year'].iloc[0]
    fund_type = supplier_data['_FundType'].iloc[0]
    country_data = supplier_data['_Country_Data'].iloc[0].copy()
    country_data['ISO'] = country_data['ShipToAddressCountry'].map(CONFIG['country_iso_mapping'])
    
    st.subheader(f"📊 {supplier_name} Performance Dashboard - {year} ({fund_type})")
    
    # Summary cards
    total_amount = supplier_data['_Total_Amount'].iloc[0]
    total_freight = supplier_data['_Total_Freight'].iloc[0]
    overall_ratio = (total_freight / total_amount * 100) if total_amount > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Product Spend", f"${total_amount:,.2f}")
    with col2:
        st.metric("Total Freight Costs", f"${total_freight:,.2f}")
    with col3:
        st.metric("Overall Freight Ratio", f"{overall_ratio:.1f}%")

    # Tab system
    tab1, tab2, tab3 = st.tabs(["🌍 Country Analysis", "📦 Freight Insights", "✈️ Shipping Methods"])
    
    # Tab 1: Country bar chart
    with tab1:
        st.markdown("### Product Spend by Country")
        sort_order = st.radio("Sort Order", ["Descending", "Ascending"], horizontal=True, index=0)
        sorted_data = country_data.sort_values(
            'Amount per PO Line',
            ascending=(sort_order == "Ascending")
        )
        
        fig1 = px.bar(
            sorted_data,
            x='ShipToAddressCountry',
            y='Amount per PO Line',
            color='Amount per PO Line',
            color_continuous_scale='teal',
            labels={'ShipToAddressCountry': 'Country', 'Amount per PO Line': 'Total Spend ($)'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    # Tab 2: Map visualization
    with tab2:
        st.markdown(f"### Freight Ratio by Country (Threshold: ≥{ratio_threshold}%)")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            country_data['Ratio_Category'] = country_data['Freight Ratio (%)'].apply(
                lambda x: 'High' if x >= ratio_threshold else 'Normal'
            )
            
            fig2 = px.choropleth(
                country_data.dropna(subset=['ISO']),
                locations="ISO",
                color="Ratio_Category",
                color_discrete_map={'High': 'orange', 'Normal': 'blue'},
                hover_name="ShipToAddressCountry",
                hover_data={
                    'Amount per PO Line': ':$,.2f',
                    'Freight per APO': ':$,.2f',
                    'Freight Ratio (%)': ':.1f%'
                },
                projection="natural earth",
                scope="south america",
                height=600
            )

            fig2.update_geos(
                showcountries=True,
                countrycolor="lightgray",
                showocean=True,
                oceancolor="lightblue",
                lataxis_range=[-60, 35],
                lonaxis_range=[-120, -30]
            )
            
            fig2.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                legend_title_text='Ratio Status'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.markdown(f"**Alerts (≥{ratio_threshold}%)**")
            high_ratio = country_data[country_data['Freight Ratio (%)'] >= ratio_threshold]
            
            if not high_ratio.empty:
                high_ratio = high_ratio.sort_values('Freight Ratio (%)', ascending=False)
                for _, row in high_ratio.iterrows():
                    st.markdown(f"""
                    <div style="
                        background-color: #fff3f3;
                        padding: 10px;
                        border-radius: 5px;
                        margin-bottom: 10px;
                        border-left: 4px solid #ff4b4b;
                    ">
                        <b>{row['ShipToAddressCountry']}</b><br>
                        Ratio: <b>{row['Freight Ratio (%)']:.1f}%</b><br>
                        Spend: ${row['Amount per PO Line']:,.2f}<br>
                        Freight: ${row['Freight per APO']:,.2f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No ratios above {ratio_threshold}%")
            
            unmapped = country_data[country_data['ISO'].isna()]
            if not unmapped.empty:
                st.warning(f"Data missing for: {', '.join(unmapped['ShipToAddressCountry'].unique())}")
    
    # Tab 3: Shipping pie chart
    with tab3:
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
                
        except Exception:
            st.warning("Shipping data not available")
    
    # Products section
    st.markdown("---")
    st.markdown("### 🛒 Products Delivered")
    products = supplier_data['Products Delivered'].iloc[0].split('\n')
    filtered_products = [p for p in products if 'discount' not in p.lower() and p.strip()]
    
    if filtered_products:
        st.markdown('\n'.join(filtered_products))
    else:
        st.info("No non-discount products found")
        with st.expander("Show full product list"):
            st.markdown(supplier_data['Products Delivered'].iloc[0])

def show_help():
    """Show help documentation in sidebar"""
    with st.sidebar:
        st.markdown("---")
        with st.expander("ℹ️ Help & Documentation", expanded=False):
            st.markdown("""
            ### PAHO Spend Analysis Tool Guide
            
            **Purpose**: Analyze supplier spending patterns with focus on freight costs.
            
            **Features**:
            - Supplier performance dashboards
            - Freight ratio analysis
            - Shipping method breakdowns
            - Country-level spending insights
            - Configurable supplier lists
            - Adjustable freight ratio thresholds
            
            **How To Use**:
            1. Upload your PAHO spend data file
            2. Select year(s) to analyze
            3. Choose a fund type (Revolving/Strategic)
            4. Select a supplier to analyze
            5. Adjust settings in the sidebar as needed
            6. Explore the interactive dashboard
            7. Download the full report
            """)

def main():
    """Main application function"""
    st.title("PAHO Spend Analysis Dashboard")
    
    # Initialize session state
    if 'custom_revolving_suppliers' not in st.session_state:
        st.session_state.custom_revolving_suppliers = CONFIG['default_revolving_suppliers'].copy()
    if 'custom_strategic_suppliers' not in st.session_state:
        st.session_state.custom_strategic_suppliers = CONFIG['default_strategic_suppliers'].copy()
    if 'ratio_threshold' not in st.session_state:
        st.session_state.ratio_threshold = 50
    
    show_help()
    
    uploaded_file = st.file_uploader("Upload PAHO Spend Data", type="xlsx")
    
    if uploaded_file:
        with st.spinner("Processing data..."):
            try:
                df = load_data(uploaded_file)
                validate_data(df)
                
                all_revolving = sorted(df[df['FundType'] == 'Revolving Fund']['Supplier'].unique().tolist())
                all_strategic = sorted(df[df['FundType'] == 'Strategic Fund']['Supplier'].unique().tolist())
                
                with st.sidebar:
                    st.markdown("---")
                    available_years = sorted(df['YearReceipt'].unique(), reverse=True)
                    selected_year = st.selectbox("Select Year:", options=available_years, index=0)
                    
                    st.session_state.ratio_threshold = st.number_input(
                        "High Freight Ratio Threshold (%)",
                        min_value=0, max_value=100, value=st.session_state.ratio_threshold,
                        step=1,
                        help="Threshold for highlighting high freight ratios"
                    )
                    
                    with st.expander("⚙️ Manage Supplier Lists"):
                        st.markdown("### Revolving Fund Suppliers")
                        revolving_selected = st.multiselect(
                            "Select suppliers:",
                            options=all_revolving,
                            default=[s for s in st.session_state.custom_revolving_suppliers if s in all_revolving]
                        )
                        
                        st.markdown("### Strategic Fund Suppliers")
                        strategic_selected = st.multiselect(
                            "Select suppliers:",
                            options=all_strategic,
                            default=[s for s in st.session_state.custom_strategic_suppliers if s in all_strategic]
                        )
                        
                        if st.button("Apply Selections"):
                            st.session_state.custom_revolving_suppliers = revolving_selected
                            st.session_state.custom_strategic_suppliers = strategic_selected
                        
                        if st.button("Reset to Defaults"):
                            st.session_state.custom_revolving_suppliers = [s for s in CONFIG['default_revolving_suppliers'] if s in all_revolving]
                            st.session_state.custom_strategic_suppliers = [s for s in CONFIG['default_strategic_suppliers'] if s in all_strategic]
                
                final_report = build_final_report(
                    df,
                    [selected_year],
                    st.session_state.custom_revolving_suppliers,
                    st.session_state.custom_strategic_suppliers
                )
                
                excel_data = generate_excel(
                    final_report=final_report,
                    ratio_threshold=st.session_state.ratio_threshold,
                    selected_year=selected_year
                )
                st.download_button(
                    label="📥 Download Full Report",
                    data=excel_data,
                    file_name=f"PAHO_Supplier_Report_{selected_year}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                fund_type = st.radio(
                    "Select Fund Type:",
                    ["Revolving Fund", "Strategic Fund"],
                    horizontal=True
                )
                
                vendors = final_report[final_report['_FundType'] == fund_type]['Supplier'].unique()
                if len(vendors) == 0:
                    st.warning(f"No {fund_type} suppliers found")
                    return
                
                selected_supplier = st.selectbox(f"Select {fund_type} Supplier:", options=sorted(vendors))
                supplier_data = final_report[(final_report['Supplier'] == selected_supplier) & (final_report['_FundType'] == fund_type)]
                
                if not supplier_data.empty:
                    create_supplier_dashboard(supplier_data, st.session_state.ratio_threshold)
                else:
                    st.error(f"No data found for {selected_supplier}")
                
                with st.expander(f"🔍 View All {fund_type} Suppliers Data"):
                    fund_df = final_report[final_report['_FundType'] == fund_type].drop(
                        columns=['_Ratio_Values', '_FundType', '_Country_Data'])
                    st.dataframe(fund_df, height=400, use_container_width=True, hide_index=True)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
