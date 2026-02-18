import pandas as pd
import glob
import os

RAW_DIR = "/data4t/projects/fs/data_raw"
SAMPLE_SHEET = "/data4t/projects/fs/data_raw/gdc_sample_sheet.2026-02-16.tsv"
OUTPUT_DIR = "/data4t/projects/fs/data_processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "brca_ml_matrix.csv")
LABEL_FILE = os.path.join(OUTPUT_DIR, "labels.csv")

def main():
    print("Loading sample metadata...")
    ss = pd.read_csv(SAMPLE_SHEET, sep='\t')
    ss.columns = ss.columns.str.strip()

    # Create a unique ID for every file: CaseID_FileID
    # This prevents the 136 'repeats' from being overwritten
    ss['unique_id'] = ss['Case ID'].astype(str) + "_" + ss['File ID'].astype(str)
    
    # Map Folder (File ID) to our new Unique ID and Tissue Type
    mapping = ss.set_index('File ID')[['unique_id', 'Tissue Type']].to_dict('index')

    files = glob.glob(os.path.join(RAW_DIR, "**/*.rna_seq.augmented_star_gene_counts.tsv"), recursive=True)
    print(f"Found {len(files)} files on disk.")

    data_dict = {}
    label_list = []

    for i, fpath in enumerate(files):
        folder_id = os.path.basename(os.path.dirname(fpath))
        
        if folder_id in mapping:
            info = mapping[folder_id]
            uid = info['unique_id']
            tissue = info['Tissue Type']
            
            # Read data
            df = pd.read_csv(fpath, sep='\t', skiprows=1, usecols=['gene_name', 'tpm_unstranded'])
            
            # Clean: handle potential NaNs and stats lines
            df = df.dropna(subset=['gene_name'])
            df['gene_name'] = df['gene_name'].astype(str)
            df = df.loc[~df['gene_name'].str.startswith('N_')]
            
            # Save data and record the label (Tumor=1, Normal=0)
            data_dict[uid] = dict(zip(df['gene_name'], df['tpm_unstranded']))
            label_list.append({'unique_id': uid, 'target': 1 if 'Tumor' in tissue else 0})
        
        if (i+1) % 200 == 0:
            print(f"Processed {i+1}/{len(files)} samples...")

    print("Building Final Matrix (1231 rows expected)...")
    master_df = pd.DataFrame.from_dict(data_dict, orient='index')
    
    print("Building Label File...")
    label_df = pd.DataFrame(label_list)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Saving matrix to {OUTPUT_FILE} (This might take a minute)...")
    master_df.to_csv(OUTPUT_FILE)
    
    print(f"Saving labels to {LABEL_FILE}...")
    label_df.to_csv(LABEL_FILE, index=False)
    
    print(f"DONE!")
    print(f"Matrix Final Shape: {master_df.shape}")
    print(f"Total Samples saved: {len(label_df)}")

if __name__ == "__main__":
    main()
