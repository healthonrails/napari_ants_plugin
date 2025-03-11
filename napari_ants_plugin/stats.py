import argparse
import os
import glob
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from io import StringIO

# Set up logging configuration.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def extract_metadata(filepath):
    """
    Extracts the brain ID and channel from the CSV file name.
    Expected filename format:
      cell_counts_<brain_id>_<channel>_...csv
    For example:
      cell_counts_MF1_133F_W_BS_640_kim_mouse_25um.csv
    Returns:
      brain_id: e.g. "MF1_133F_W_BS"
      channel: e.g. "640"
    """
    base = os.path.basename(filepath)
    pattern = r"cell_counts_([^_]+(?:_[^_]+)*)_(\d+)_.*\.csv"
    match = re.match(pattern, base)
    if match:
        brain_id = match.group(1)
        channel = match.group(2)
        return brain_id, channel
    else:
        logging.warning(f"Failed to extract metadata from filename: {base}")
        return None, None


def save_text(output_path, filename, content):
    """
    Saves text content to a file in the output_path directory.
    """
    full_path = os.path.join(output_path, filename)
    with open(full_path, "w") as f:
        f.write(content)
    logging.info(f"Saved {filename}")


def plot_violin_for_regions(data, output_folder, regions, filename, title):
    """
    Creates a violin plot for a provided list of brain regions.
    Only regions present in the data will be plotted.
    """
    # Filter the list to only include regions present in the data.
    available = set(data['brain_region'].unique())
    regions_to_plot = [r for r in regions if r in available]
    missing = set(regions) - set(regions_to_plot)
    if missing:
        logging.warning(f"No data found for regions: {missing}")
    if not regions_to_plot:
        logging.error("No valid regions to plot.")
        return

    violin_data = [data.loc[data["brain_region"] == region, "cell_count"].values
                   for region in regions_to_plot]

    plt.figure(figsize=(10, 6))
    plt.violinplot(violin_data, showmedians=True)
    plt.xticks(ticks=np.arange(1, len(regions_to_plot) + 1),
               labels=regions_to_plot, rotation=45, ha="right")
    plt.ylabel("Cell Count")
    plt.title(title)
    plt.tight_layout()

    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path)
    logging.info(f"Saved violin plot for regions to {output_path}")
    plt.close()


def plot_violin_by_brain(data, output_folder):
    """
    Creates a violin plot for cell count distribution by brain (brain_id).
    """
    brain_groups = data.groupby("brain_id")["cell_count"]
    brain_order = brain_groups.median().sort_values().index.tolist()
    brain_data = [data.loc[data["brain_id"] == brain,
                           "cell_count"].values for brain in brain_order]

    plt.figure(figsize=(12, max(6, len(brain_order)*0.3)))
    plt.violinplot(brain_data, showmedians=True)
    plt.xticks(ticks=np.arange(1, len(brain_order)+1),
               labels=brain_order, rotation=45, ha="right")
    plt.ylabel("Cell Count")
    plt.title("Violin Plot: Cell Count Distribution by Brain")
    plt.tight_layout()

    output_file = os.path.join(output_folder, "violin_by_brain.png")
    plt.savefig(output_file)
    logging.info(f"Saved violin plot by brain to {output_file}")
    plt.close()


def plot_violin_by_channel(data, output_folder):
    """
    Creates a violin plot for cell count distribution by channel.
    """
    channel_groups = data.groupby("channel")["cell_count"]
    channel_order = channel_groups.median().sort_values().index.tolist()
    channel_data = [data.loc[data["channel"] == ch,
                             "cell_count"].values for ch in channel_order]

    plt.figure(figsize=(8, 6))
    plt.violinplot(channel_data, showmedians=True)
    plt.xticks(ticks=np.arange(1, len(channel_order)+1), labels=channel_order)
    plt.ylabel("Cell Count")
    plt.xlabel("Channel")
    plt.title("Violin Plot: Cell Count Distribution by Channel")
    plt.tight_layout()

    output_file = os.path.join(output_folder, "violin_by_channel.png")
    plt.savefig(output_file)
    logging.info(f"Saved violin plot by channel to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process cell count CSV files for mouse brains and save results for research papers."
    )
    parser.add_argument("folder", help="Folder path containing CSV files.")
    parser.add_argument(
        "--pattern",
        default="cell_counts_*.csv",
        help="CSV file pattern to search for (default: cell_counts_*.csv)."
    )
    parser.add_argument(
        "--output",
        default="stats_results",
        help="Output folder to save analysis results (default: 'stats_results')."
    )
    parser.add_argument(
        "--top_n", type=int, default=25,
        help="Plot top N brain regions by median cell count (default: 25)."
    )
    parser.add_argument(
        "--regions", type=str, default="MOB,AOB,OLF,OT",
        help="Comma-separated list of brain regions to plot as default. Default: 'MOB,AOB,OLF,OT'."
    )
    args = parser.parse_args()

    base_folder = args.folder
    file_pattern = args.pattern
    output_folder = args.output

    # Create the output folder if it doesn't exist.
    os.makedirs(output_folder, exist_ok=True)

    # Recursively find CSV files matching the given pattern.
    search_pattern = os.path.join(base_folder, '**', file_pattern)
    csv_files = glob.glob(search_pattern, recursive=True)
    logging.info(
        f"Found {len(csv_files)} CSV files matching pattern '{file_pattern}' in folder '{base_folder}'.")

    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Rename columns for consistency: 'acronym' -> 'brain_region'
            df = df.rename(
                columns={'acronym': 'brain_region', 'cell_count': 'cell_count'})
            # Exclude rows where the brain region is the root (e.g., "grey" or "root").
            df = df[~df['brain_region'].str.lower().isin(['grey', 'root'])]
            # Extract metadata from the file name.
            brain_id, channel = extract_metadata(file)
            df['brain_id'] = brain_id if brain_id is not None else "Unknown"
            df['channel'] = channel if channel is not None else "Unknown"
            dataframes.append(df)
            logging.info(f"Processed file: {file} (excluded root regions)")
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")

    if not dataframes:
        logging.error("No CSV files were read successfully.")
        raise ValueError("No CSV files were read successfully.")

    # Combine all individual DataFrames into one master DataFrame.
    data = pd.concat(dataframes, ignore_index=True)

    # Save combined dataset information.
    save_text(output_folder, "combined_head.txt", data.head().to_string())
    info_buffer = StringIO()
    data.info(buf=info_buffer)
    save_text(output_folder, "combined_info.txt", info_buffer.getvalue())
    save_text(output_folder, "combined_description.txt",
              data.describe().to_string())

    # ---------------------------------------
    # Descriptive Statistics
    # ---------------------------------------
    save_text(output_folder, "brain_stats.txt",
              data.groupby("brain_id")["cell_count"].describe().to_string())
    save_text(output_folder, "region_stats.txt",
              data.groupby("brain_region")["cell_count"].describe().to_string())
    save_text(output_folder, "channel_stats.txt",
              data.groupby("channel")["cell_count"].describe().to_string())

    # ---------------------------------------
    # Data Visualizations (using violin plots)
    # ---------------------------------------
    plot_violin_by_brain(data, output_folder)
    plot_violin_by_channel(data, output_folder)

    # Produce violin plot for top N brain regions.
    region_summary = data.groupby("brain_region")["cell_count"].median()
    top_regions = region_summary.sort_values(
        ascending=False).head(args.top_n).index.tolist()
    plot_violin_for_regions(
        data,
        output_folder,
        top_regions,
        f"violin_top_{args.top_n}_regions.png",
        f"Violin Plot of Cell Count Distributions for Top {args.top_n} Brain Regions"
    )

    # Produce violin plot for default olfactory-related regions.
    default_regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    plot_violin_for_regions(
        data,
        output_folder,
        default_regions,
        "violin_default_regions.png",
        "Violin Plot of Cell Count Distributions for Default (Olfactory-related) Brain Regions"
    )

    # ---------------------------------------
    # Statistical Analysis: ANOVA
    # ---------------------------------------
    anova_brain_model = smf.ols('cell_count ~ C(brain_id)', data=data).fit()
    anova_brain_table = sm.stats.anova_lm(anova_brain_model, typ=2)
    save_text(output_folder, "anova_brain.txt", anova_brain_table.to_string())

    anova_region_model = smf.ols(
        'cell_count ~ C(brain_region)', data=data).fit()
    anova_region_table = sm.stats.anova_lm(anova_region_model, typ=2)
    save_text(output_folder, "anova_region.txt",
              anova_region_table.to_string())

    anova_channel_model = smf.ols('cell_count ~ C(channel)', data=data).fit()
    anova_channel_table = sm.stats.anova_lm(anova_channel_model, typ=2)
    save_text(output_folder, "anova_channel.txt",
              anova_channel_table.to_string())


if __name__ == "__main__":
    main()
