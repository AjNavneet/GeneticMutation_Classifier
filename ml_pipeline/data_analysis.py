import numpy as np
import matplotlib.pyplot as plt
import cv2

class DataAnalysis:
    def __init__(self, file_path, train_data) -> None:
        '''Initialize the class with a file path to save output figures and training data.'''
        self.train_data = train_data
        self.file_path = file_path

    def frequency_analysis(self, dataset, feature_name, titlename=""):
        '''Generate frequency distribution plots for a specified feature in the training data.'''
        
        # Create a title for the plot
        Title = "Frequency of " + feature_name + " in training data"
        
        # Check if the number of unique values in the feature is greater than 10
        if len(sorted(set(dataset))) > 10:
            no_shown = 10
            df = self.train_data.groupby(by=feature_name)
            
            # Generate a bar plot showing the top 10 most frequent values
            df_plot = df.count().sort_values(by='ID', ascending=False).head(no_shown).plot(
                kind='bar', y='ID', ylabel='Frequency', xlabel=feature_name, color='purple', title=Title
            )
            
            # Save the plot as an image file
            df_plot.figure.savefig(self.file_path + '/' + feature_name + '.png')
            plt.close(df_plot.figure)
        else:
            # If there are fewer than 10 unique values, generate a histogram
            plt.title(Title)
            plt.xlabel(feature_name)
            plt.ylabel('Frequency')
            plt.hist(dataset, bins=9, align='mid', color='purple', edgecolor='black', rwidth=0.7)
            
            # Save the histogram as an image file
            plt.savefig(self.file_path + '/' + feature_name + '.png')
            plt.close()

    def distribution_analysis(self, y_train, y_val, y_test):
        '''Generate histograms to visualize the distribution of classes in the training, validation, and testing data.'''
        
        # Create a figure with three subplots
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        fig.tight_layout(pad=5.0)
        
        # Prepare the datasets and names
        datasets = [y_train, y_val, y_test]
        dataset_names = ['training', 'validation', 'testing']
        
        # Generate histograms for each dataset and add labels
        for ind, (data, name) in enumerate(zip(datasets, dataset_names)):
            ax[ind].set_title("Frequency/distribution of classes in " + name + " data")
            ax[ind].set_xlabel('Classes')
            ax[ind].set_ylabel('Frequency')
            ax[ind].hist(sorted(data), bins=9, align='mid', color='purple', edgecolor='black', rwidth=0.7)
        
        # Save the figure as an image file
        plt.savefig(self.file_path + '/distribution.png')
        plt.close()
