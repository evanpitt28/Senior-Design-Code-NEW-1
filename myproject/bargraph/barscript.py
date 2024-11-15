import matplotlib.pyplot as plt
import numpy as np
import os
from django.conf import settings


# Sample data, will configure to take inputs from ML/proprocessing
#categories = ['Stroke', 'Seizure', 'N/A']
#values = [80,15,5]
#confidence_intervals = [5,5,5]  # Sample confidence intervals

def create_bar_graph_with_ci():
    plt.clf()  # Clear the current figure
    plt.close('all')  # Close all open figures

    categories = ['Stroke', 'Seizure', 'N/A']
    values = [98,0.5,1.5]
    confidence_intervals = [2,2,2]

    # Set up the figure and plot
    fig, ax = plt.subplots()

    # Bar chart
    bars = ax.bar(categories, values, yerr=confidence_intervals, capsize=5, label="Value")

     # Label each bar with its value as a percentage
    for bar, value, ci in zip(bars, values, confidence_intervals):
        # Position of the text
        y_position = bar.get_height() + ci + 1  # Adjust position above the bar + confidence interval
        # Add text label on each bar, showing the value and CI in percentage
        ax.text(bar.get_x() + bar.get_width() / 2, y_position, f"{value}% ± {ci}%", 
                ha='center', va='bottom', fontsize=10, color='black')


    # Labels and title
    ax.set_xlabel('Potential Diagnoses')
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Diagnosis Graph with Confidence Intervals')

    # Save the plot as an image
    output_path = os.path.join(settings.BASE_DIR,'static', 'bar_graph_ci.png')
    print(f"Saving image to: {output_path}")
    plt.savefig(output_path)  # Save to Django static folder
    
def create_bar_graph_with_ci():
    plt.clf()  # Clear the current figure
    plt.close('all')  # Close all open figures

    categories = ['Stroke', 'Seizure', 'N/A']
    values = [5,90,5]
    confidence_intervals = [2.5,1,1]

    # Set up the figure and plot
    fig, ax = plt.subplots()

    # Bar chart
    bars = ax.bar(categories, values, yerr=confidence_intervals, capsize=5, label="Value")

     # Label each bar with its value as a percentage
    for bar, value, ci in zip(bars, values, confidence_intervals):
        # Position of the text
        y_position = bar.get_height() + ci + 1  # Adjust position above the bar + confidence interval
        # Add text label on each bar, showing the value and CI in percentage
        ax.text(bar.get_x() + bar.get_width() / 2, y_position, f"{value}% ± {ci}%", 
                ha='center', va='bottom', fontsize=10, color='black')


    # Labels and title
    ax.set_xlabel('Potential Diagnoses')
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Diagnosis Graph with Confidence Intervals')

    # Save the plot as an image
    output_path = os.path.join(settings.BASE_DIR,'static', 'bar_graph_ci_example.png')
    print(f"Saving image to: {output_path}")
    plt.savefig(output_path)  # Save to Django static folder