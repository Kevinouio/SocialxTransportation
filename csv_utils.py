import os
import pandas as pd

def update_street_statistics_csv(data, rumor_street, csv_filename="street_crossings.csv"):
    if os.path.exists(csv_filename):
        existing_df = pd.read_csv(csv_filename)
        existing_columns = existing_df.columns.tolist()
        run_number = len(existing_columns)
        new_run_col = f"Run {run_number}"
        new_data_df = pd.DataFrame({"Street Names & Edge IDs": list(data.keys()), new_run_col: list(data.values())})
        merged_df = existing_df.merge(new_data_df, on="Street Names & Edge IDs", how="outer")
    else:
        merged_df = pd.DataFrame({"Street Names & Edge IDs": list(data.keys()), "Run 1": list(data.values())})

    rumor_row = pd.DataFrame({"Street Names & Edge IDs": ["Rumor Injected"], f"Run {len(merged_df.columns) - 1}": rumor_street}, index=[len(merged_df)])
    merged_df = pd.concat([merged_df, rumor_row], ignore_index=True)
    merged_df.to_csv(csv_filename, index=False)
    return merged_df
