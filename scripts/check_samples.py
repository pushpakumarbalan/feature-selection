import pandas as pd

SAMPLE_SHEET = "/data4t/projects/fs/data_raw/gdc_sample_sheet.2026-02-16.tsv"

def main():
    # Reading with sep='\t' and stripping whitespace from column names
    ss = pd.read_csv(SAMPLE_SHEET, sep='\t')
    ss.columns = ss.columns.str.strip() 
    
    print("--- Verified Columns in Sample Sheet ---")
    print(list(ss.columns))
    
    print("\n--- Dataset Breakdown ---")
    print(f"Total Files (Folders): {len(ss)}")
    
    if 'Case ID' in ss.columns:
        print(f"Unique Patients (Cases): {ss['Case ID'].nunique()}")
    
    # Try to find the sample type column even if name varies slightly
    # GDC usually uses 'Sample Type' or 'sample_type'
    st_col = [c for c in ss.columns if 'Sample Type' in c or 'sample_type' in c]
    
    if st_col:
        col_name = st_col[0]
        print(f"\n--- Distribution of {col_name} ---")
        print(ss[col_name].value_counts())
        
        # Check for matching pairs (Patients with both Tumor and Normal)
        if 'Case ID' in ss.columns:
            counts = ss.groupby('Case ID')[col_name].nunique()
            pairs = (counts > 1).sum()
            print(f"\nPatients with Matched Pairs (e.g., Tumor + Normal): {pairs}")
    else:
        print("\nERROR: Could not find a 'Sample Type' column.")

if __name__ == "__main__":
    main()