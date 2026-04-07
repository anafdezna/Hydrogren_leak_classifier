# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 11:05:40 2026

@author: anafd
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def plot_error_distribution(distances, counts, tank_spacing=4.25):
    """
    Reproduces a high-quality, paper-ready Error Distribution bar chart.
    Optimized for publication with larger fonts and professional styling.
    """
    
    # 1. Global Styling for Academic Papers
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.dpi': 300  # High resolution for print
    })
    
    percentages = counts 
    
    # Professional Palette: Muted Sage Green and Terracotta Red
    safe_color = '#6ab04c' 
    distant_color = '#eb4d4b'
    colors = [safe_color] + [distant_color] * (len(distances) - 1)
    
    # 2. Create the figure
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Plot the bars with slight transparency and cleaner edges
    bars = ax.bar(distances, percentages, color=colors, edgecolor='#2f3640', 
                  linewidth=1.2, width=0.65, alpha=0.9)
    
    # 3. Styling the Axes
    ax.set_ylabel('Percentage of misclassifications (%)', labelpad=15)
    ax.set_xlabel(f'Error distance (m)\n' + r'$\mathit{[Tank\ spacing\ =\ ' + f'{tank_spacing}' + r'm]}$', 
                  fontsize=15, labelpad=15)
    
    # Clean up the frame (Despine)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Set y-limit with breathing room
    ax.set_ylim(0, 100) 
    
    # Subtle horizontal grid
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)
    ax.set_axisbelow(True)
    
    # 4. Add Value Labels with increased font weight
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{height:.1f}%', ha='center', va='bottom', 
                fontsize=13, fontweight='bold', color='#2f3640')

    # 5. Legend and Stats (Correcting Overlap)
    legend_elements = [
        Line2D([0], [0], color=safe_color, lw=10, label='Neighbor errors (safe)'),
        Line2D([0], [0], color=distant_color, lw=10, label='Distant errors')
    ]
    
    # Place Legend in a clean white box
    ax.legend(handles=legend_elements, loc='upper right', 
              frameon=True, facecolor='white', framealpha=1, edgecolor='#dcdde1')
    
    # Stats Summary Box (Positioned to avoid legend)
    total_val = int(sum(counts))
    stats_text = (
        # r"$\mathbf{Summary\ Statistics}$" + "\n"
        f"Total missclassifications: 252\n"
        f"Neighbor errors: {percentages[0]:.1f}%\n"
        f"Critical errors: {sum(percentages[1:]):.1f}%"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor='#f5f6fa', alpha=1, edgecolor='#dcdde1')
    ax.text(0.97, 0.72, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props, linespacing=1.5)

    plt.tight_layout()
    return fig

# --- Run Example ---
if __name__ == "__main__":
    dist_labels = ['4.25m', '8.50m', '12.75m', '17.00m', '21.25m']
    error_values = [79.8, 0.8, 11.9, 4.4, 3.2]
    
    fig = plot_error_distribution(dist_labels, error_values)
    # To save for your paper, you would use:
    plt.savefig('Error_distribution_revised.png', dpi=500, bbox_inches='tight')
    plt.show()
    
    
    
import os
foldername = '50secsSim_Windows_06Feb20266TANKS10nfps50steps_60000epoch_1024batch1e-05LR'
test_report= np.load(os.path.join(foldername, "TrainClassification_Report.npy"), allow_pickle = True)
