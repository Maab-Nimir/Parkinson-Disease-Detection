import tensorflow as tf
import matplotlib.pyplot as plt

def plot_training(event_file):
    # Extract data from the event file
    step = []  # Store the global steps
    values = []  # Store the values (e.g., loss or accuracy)
    tags = []  # Store the tags (e.g., "train_loss", "validation_accuracy")
    
    for summary in tf.compat.v1.train.summary_iterator(event_file):
        for value in summary.summary.value:
            # You can filter out the tags you're interested in (e.g., "train_loss", "accuracy")
            if value.HasField('simple_value'):  # Make sure it is a scalar value
                tags.append(value.tag)
                values.append(value.simple_value)
                step.append(summary.step)
    
    # Plot the loss
    plt.figure(figsize=(10, 6))
    
    # Example: Plotting a specific tag (e.g., "train_loss")
    for tag in set(tags):  # We loop through each unique tag
        # print(tag)
        if tag == 'error/valid':
            continue
        if tag == 'Epoch':
            continue
        if tag == 'acc/train':
            continue
        if tag == 'acc/valid':
            continue
        tag_indices = [i for i, t in enumerate(tags) if t == tag]
        tag_steps = [step[i] for i in tag_indices]
        tag_values = [values[i] for i in tag_indices]
        
        plt.plot(tag_steps, tag_values, label=tag)  # Plot each tag with its respective values
    
    # Customize plot
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Loss during training')
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.show()
    
    # Plot the acc
    plt.figure(figsize=(10, 6))
    
    # Example: Plotting a specific tag (e.g., "train_loss")
    for tag in set(tags):  # We loop through each unique tag
        # print(tag)
        if tag == 'error/valid':
            continue
        if tag == 'Epoch':
            continue
        if tag == 'loss/train':
            continue
        if tag == 'loss/valid':
            continue
        tag_indices = [i for i, t in enumerate(tags) if t == tag]
        tag_steps = [step[i] for i in tag_indices]
        tag_values = [values[i] for i in tag_indices]
        
        plt.plot(tag_steps, tag_values, label=tag)  # Plot each tag with its respective values
    
    # Customize plot
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Accuracy during training')
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.show()