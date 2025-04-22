import seaborn as sns
import matplotlib.pyplot as plt

def plot_cm(cm, title='Confusion Matrix'):
    labels = ['PD', 'HC']
    
    # Plotting
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar=False)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()
