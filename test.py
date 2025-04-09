import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Configuration ---
csv_filepath = 'combined_voltage_log.csv'
nodes_to_plot_count = 200 # How many nodes to plot from the top
# --- End Configuration ---

try:
    # Read the CSV file
    df = pd.read_csv(csv_filepath, header=0)

    # Check if 'Bus' column exists
    if 'Bus' not in df.columns:
        print("Error: 'Bus' column not found in the header row (first row).")
        print("Actual columns found:", df.columns.tolist())
        exit()

    # Skip the first two data rows ('MainPowerGrid', 'LocalSubstation')
    if len(df) > 1:
        df = df.iloc[2:].copy()
    else:
        print("Warning: CSV file has less than 3 rows (including header). Cannot skip first two data rows.")
        pass

    # Set 'Bus' column as index
    if not df.empty:
        df = df.set_index('Bus')

        # Get voltage columns and convert to numeric
        voltage_columns = df.columns
        df[voltage_columns] = df[voltage_columns].apply(pd.to_numeric, errors='coerce')

        # --- Select only the first N nodes for plotting ---
        if len(df) < nodes_to_plot_count:
            print(f"Warning: Fewer than {nodes_to_plot_count} nodes available after skipping initial rows ({len(df)} found). Plotting all available nodes.")
            df_to_plot = df # Select all available if fewer than requested
        else:
            # Select the first 'nodes_to_plot_count' rows using iloc
            df_to_plot = df.iloc[:nodes_to_plot_count]
            print(f"Selecting first {nodes_to_plot_count} nodes for plotting: {df_to_plot.index.tolist()}")
        # ---------------------------------------------------

        if df_to_plot.empty:
             print("Error: No nodes left to plot after selection.")
             exit()

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 9))

        print(f"Processing {len(df_to_plot)} selected nodes for plotting (values clipped to [0, 1] range)...")
        nodes_plotted = 0
        # Iterate over the *selected subset* of the DataFrame
        for node_name, voltage_data in df_to_plot.iterrows():
            if voltage_data.notna().any():
                # Create a temporary Series where values are clipped to the range [0, 1]
                plot_data = voltage_data.clip(lower=0, upper=1)

                ax.plot(voltage_columns, plot_data, marker='.', linestyle='-', markersize=4, label=node_name)
                nodes_plotted += 1
            else:
                print(f"Skipping node {node_name} as it contains no valid numeric data after conversion.")

        if nodes_plotted == 0:
             print("\nWarning: None of the selected nodes had valid data to plot.")
             plt.close(fig)
             exit()

        # --- Customize Plot ---
        # Update title to reflect the number of nodes actually plotted
        ax.set_title(f'Voltage Log for First {nodes_plotted} Nodes (Values Clipped to [0, 1])', fontsize=16)
        ax.set_xlabel('Time Step / Sample Index', fontsize=12)
        ax.set_ylabel('Voltage (p.u. or normalized - Clipped)', fontsize=12)

        # Set y-axis limits firmly to slightly outside the 0-1 range
        ymin_plot = -0.02
        ymax_plot = 1.02
        ax.set_ylim(ymin_plot, ymax_plot)
        # print(f"\nNote: Y-axis fixed to [{ymin_plot:.2f}, {ymax_plot:.2f}] to show clipped 0-1 range.")

        # Set y-axis ticks for clarity within the 0-1 range
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(title="Nodes", bbox_to_anchor=(1.04, 1), loc='upper left', fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.show()

    else:
        print("Error: DataFrame is empty after skipping the first two data rows. No data to plot.")


except FileNotFoundError:
    print(f"Error: File not found at {csv_filepath}")
except pd.errors.EmptyDataError:
    print(f"Error: The file {csv_filepath} is empty or permissions issue.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()