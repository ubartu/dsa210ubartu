import pandas as pd

# 1. Load the dataset

import pandas as pd
file_path = 'simpra-saatlik-yogunluk-raporu-01-01-2025-31-01-2025-1739387506.xlsx'
# 1. Load
df = pd.read_excel(file_path)

# DEBUG: see what you actually have
print("Columns before cleaning:", df.columns.tolist())

# 2. Normalize headers
df.columns = (
    df.columns
      .str.strip()
      .str.replace('\xa0', ' ', regex=False)
)
print("Columns after stripping:", df.columns.tolist())

# 3. Drop report‐date
df = df.drop(columns=[c for c in df.columns if 'Rapor Tarihi' in c])

# 4. Rename
df = df.rename(columns={
    'Saat': 'time_slot',
    'Ortalama Hesap Tutarı': 'avg_check_amount',
    'Adisyon Sayısı': 'check_count',
    'Brüt Satış Tutarı': 'gross_sales',
    'İndirim Tutarı': 'discount_amount',
    'Net Satış Tutarı': 'net_sales',
    'KDV Hariç Satış Tutarı': 'sales_excl_vat'
})
print("Columns after renaming:", df.columns.tolist())

# 4. Parse time_slot into numeric hour (0–23)
df['hour'] = df['time_slot'].str.split('-').str[0].str.slice(0,2).astype(int)

# 5. Convert Turkish-formatted numbers to floats
def parse_turkish_number(x):
    x_str = str(x)
    return float(x_str.replace('.', '').replace(',', '.'))

numeric_cols = ['avg_check_amount', 'gross_sales', 'discount_amount', 'net_sales', 'sales_excl_vat']
for col in numeric_cols:
    df[col] = df[col].apply(parse_turkish_number)

# 6. Ensure check_count is integer
df['check_count'] = df['check_count'].astype(int)

# 7. Sort by hour and reset index
df = df.sort_values('hour').reset_index(drop=True)

# 8. Identify missing hours
all_hours = set(range(24))
present_hours = set(df['hour'])
missing_hours = sorted(list(all_hours - present_hours))
print("Missing hours in the data:", missing_hours)

# 9. Display the first 10 cleaned rows
print(df.head(10))

# At the end of your existing code, add:

# Save the cleaned DataFrame to different file formats
output_excel_path = 'cleaned_coffee_shop_data.xlsx'
output_csv_path = 'cleaned_coffee_shop_data.csv'

# Save as Excel
df.to_excel(output_excel_path, index=False)
print(f"Cleaned data saved to Excel file: {output_excel_path}")

# Save as CSV
df.to_csv(output_csv_path, index=False)
print(f"Cleaned data saved to CSV file: {output_csv_path}")