#!/usr/bin/env python

# Copyright (c) 2026 Robert Bosch GmbH and its subsidiaries.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This implementation was independently reimplemented based on the publicly
# available E2E Parking CARLA project:
#   https://github.com/qintonguav/e2e-parking-carla
#
# No original source code from the upstream project has been copied.

__author__      = "Min Hee Jo"
__copyright__   = "Copyright 2026, Robert Bosch GmbH"
__license__     = "AGPL"
__version__     = "3.0" 
__email__       = "minhee.jo@de.bosch.com"

"""
Model Comparison Bar Plot for E2E Parking Benchmark

Compares three models across six scenarios using four key metrics (TSR, NTSR, CR, TR).

Usage:
    python plot_model_comparison.py [--output OUTPUT_PATH]
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

plt.rcParams['font.family'] = 'Times New Roman'

BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# Define the models and scenarios
MODELS = ['e2e_parking_carla', 'caa_policy', 'dino_diffusion_parking']
MODEL_LABELS = ['E2E Parking', 'CAA Policy', 'DDP']

SCENARIOS = ['reproduced', 'baseline', 'misparked', 'misparked_driveout', 'misparked_follow', 'misparked_block']
SCENARIO_LABELS = ['Reproduced', 'Baseline', 'Misparked', 'Misparked\n+ Drive-Out', 'Misparked\n+ Follow', 'Misparked\n+ Block']

# Define metrics to plot (excluding TFR, NTFR, OR which are always 0)
METRICS = ['TSR', 'NTSR', 'CR', 'TR']

# Color palette for models
COLORS = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red


def load_results_from_csv(results_dir: str, model: str, scenario: str):
    """
    Load mean and std from CSV files.
    
    Expected directory structure:
    results_dir/
        {model}/
            {scenario}/
                result_mean.csv
                result_std.csv
    """
    base_path = Path(results_dir) / model / scenario
    mean_path = base_path / 'result_mean.csv'
    std_path = base_path / 'result_std.csv'
    
    if mean_path.exists() and std_path.exists():
        mean_df = pd.read_csv(mean_path, index_col=0)
        std_df = pd.read_csv(std_path, index_col=0)
        
        # Get the 'Avg' row
        mean_values = mean_df.loc['Avg']
        std_values = std_df.loc['Avg']
        
        return {metric: (mean_values[metric], std_values[metric]) for metric in METRICS}
    
    return None


def get_sample_data():
    """
    Returns sample/placeholder data structure.
    Replace this with actual data loading or fill in the values manually.
    
    Format: data[model][scenario][metric] = (mean, std)
    """
    # Initialize data structure with placeholder NaN values
    data = {model: {scenario: {metric: (np.nan, np.nan) for metric in METRICS} 
                    for scenario in SCENARIOS} 
            for model in MODELS}
    
    return data


def create_comparison_plot(data, output_path=None, figsize=(20, 5)):
    """
    Create a comprehensive bar plot comparing all models across scenarios.
    
    Args:
        data: Dictionary with structure data[model][scenario][metric] = (mean, std)
        output_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    n_scenarios = len(SCENARIOS)
    
    # Create subplots - one for each metric
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    axes = axes.flatten()
    # or just plt.show() in scripts
    # Bar width and positions
    bar_width = 0.25
    x = np.arange(n_scenarios)
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        for model_idx, (model, model_label, color) in enumerate(zip(MODELS, MODEL_LABELS, COLORS)):
            means = []
            stds = []
            
            for scenario in SCENARIOS:
                mean_val, std_val = data[model][scenario][metric]
                means.append(mean_val)
                stds.append(std_val)
            
            # Convert to numpy arrays for plotting
            means = np.array(means)
            stds = np.array(stds)
            
            # Position bars
            pos = x + (model_idx - 1) * bar_width
            
            # Create bars with error bars
            bars = ax.bar(pos, means, bar_width, 
                         label=model_label if idx == 0 else "",
                         color=color, alpha=0.8,
                         yerr=stds, capsize=3, 
                         error_kw={'elinewidth': 1, 'capthick': 1})

        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_LABELS, fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add metric name as title
        ax.set_title(metric, fontsize=20)
        
        # Adjust y-axis limits based on metric type
        ax.set_ylim(0, 105)  # Percentage metrics
    
    # Add legend
    fig.legend(MODEL_LABELS, loc='lower center', ncol=3, fontsize=14, 
               bbox_to_anchor=(0.5, 0.001), frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {output_path}")
    
    # plt.show()
    
    return fig


def print_data_table(data):
    """Print a formatted table of all data for verification."""
    print("\n" + "="*120)
    print("DATA SUMMARY (Mean ± Std)")
    print("="*120)
    
    for model in MODELS:
        print(f"\n{model.upper()}")
        print("-" * 120)
        
        # Header
        header = f"{'Scenario':<20}"
        for metric in METRICS:
            header += f"{metric:>14}"
        print(header)
        print("-" * 120)
        
        for scenario in SCENARIOS:
            row = f"{scenario:<20}"
            for metric in METRICS:
                mean_val, std_val = data[model][scenario][metric]
                if np.isnan(mean_val):
                    row += f"{'N/A':>14}"
                else:
                    row += f"{mean_val:.2f} ± {std_val:.2f}".rjust(14)
            print(row)
    
    print("\n" + "="*120)


def main():
    parser = argparse.ArgumentParser(description='Plot model comparison for E2E Parking benchmark')
    parser.add_argument('--output', '-o', type=str, default=f'{BENCHMARK_ROOT}/plots/comparison.png',
                        help='Output path for the figure (e.g., comparison.png)')
    parser.add_argument('--results-dir', '-r', type=str, default=f'{BENCHMARK_ROOT}/results',
                        help='Directory containing results organized by model/scenario')
    parser.add_argument('--metric', '-m', type=str, default=None,
                        help='Plot only a specific metric (e.g., TSR, APE)')
    args = parser.parse_args()
    
    # Try to load from CSV files
    data = {model: {scenario: {} for scenario in SCENARIOS} for model in MODELS}
    for model in MODELS:
        for scenario in SCENARIOS:
            result = load_results_from_csv(args.results_dir, model, scenario)
            if result:
                data[model][scenario] = result
            else:
                data[model][scenario] = {metric: (np.nan, np.nan) for metric in METRICS}
    
    # Print data table
    print_data_table(data)
    
    # Create plot
    create_comparison_plot(data, args.output)


if __name__ == '__main__':
    main()
