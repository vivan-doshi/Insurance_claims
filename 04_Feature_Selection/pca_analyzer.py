import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class PCAAnalyzer:
    """
    A class for performing Principal Component Analysis (PCA) on datasets.
    
    This class provides methods to perform PCA on all variables in a dataset
    or on a subset of correlated variables.
    """
    
    def __init__(self, data):
        """
        Initialize the PCAAnalyzer with a dataset.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset to analyze.
        """
        self.data = data
        self.scaler = StandardScaler()
        self.correlation_matrix = None
        self.pca_all = None
        self.pca_correlated = None
        self.transformed_all = None
        self.transformed_correlated = None
        self.explained_variance_all = None
        self.explained_variance_correlated = None
        
    def perform_pca_all(self, n_components=None, variance_threshold=None):
        """
        Perform PCA on all variables in the dataset.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of principal components to keep.
            If None, all components are kept.
        variance_threshold : float, optional
            If provided, keep only enough components to explain this 
            much variance (e.g., 0.95 for 95% variance explained).
            If provided, this takes precedence over n_components.
            
        Returns:
        --------
        transformed_df : pandas.DataFrame
            The data transformed into the principal component space as a DataFrame.
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.data)
        
        # First fit PCA with all components to calculate explained variance
        temp_pca = PCA()
        temp_pca.fit(scaled_data)
        
        # Determine number of components based on variance threshold if provided
        if variance_threshold is not None:
            if not (0 < variance_threshold <= 1.0):
                raise ValueError("variance_threshold must be between 0 and 1")
                
            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            
            # Find how many components are needed to reach the variance threshold
            n_components_for_threshold = np.argmax(cumulative_variance >= variance_threshold) + 1
            
            print(f"Using {n_components_for_threshold} components to explain {variance_threshold*100:.1f}% of variance")
            n_components = n_components_for_threshold
        
        # Initialize PCA with determined number of components
        self.pca_all = PCA(n_components=n_components)
        
        # Fit and transform the data
        self.transformed_all = self.pca_all.fit_transform(scaled_data)
        
        # Store explained variance
        self.explained_variance_all = self.pca_all.explained_variance_ratio_
        
        # Create a DataFrame from the transformed data
        cols = [f'PC{i+1}' for i in range(self.transformed_all.shape[1])]
        transformed_df = pd.DataFrame(data=self.transformed_all, columns=cols, index=self.data.index)
        
        # Display the results
        self._display_pca_results(self.pca_all, "All Variables")
        
        return transformed_df
    
    def perform_pca_correlated(self, correlation_threshold=0.7, n_components=None, variance_threshold=None):
        """
        Perform PCA on variables that are correlated above a threshold and replace
        those correlated columns with PCA components in the original dataframe.
        
        Parameters:
        -----------
        correlation_threshold : float, default=0.7
            Threshold for correlation. Variables with absolute correlation 
            above this threshold will be included.
        n_components : int, optional
            Number of principal components to keep.
            If None, all components are kept.
        variance_threshold : float, optional
            If provided, keep only enough components to explain this 
            much variance (e.g., 0.95 for 95% variance explained).
            If provided, this takes precedence over n_components.
            
        Returns:
        --------
        result_df : pandas.DataFrame
            A dataframe with uncorrelated original columns and PCA components 
            replacing the correlated columns.
        """
        # Calculate correlation matrix if not already done
        if self.correlation_matrix is None:
            self.correlation_matrix = self.data.corr().abs()
        
        # Find highly correlated features
        # Get upper triangle of correlation matrix (excluding diagonal)
        upper = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs of correlated columns
        correlated_pairs = []
        for col in upper.columns:
            # Find columns where correlation > threshold
            high_corr_cols = upper[col][upper[col] > correlation_threshold].index.tolist()
            for high_col in high_corr_cols:
                correlated_pairs.append((col, high_col))
        
        # Get unique columns from pairs
        correlated_columns = list(set([col for pair in correlated_pairs for col in pair]))
        
        if not correlated_columns:
            print(f"No variables found with correlation above {correlation_threshold}")
            return self.data.copy()  # Return original data if no correlated variables
        
        print(f"Found {len(correlated_columns)} variables with high correlation (>{correlation_threshold}):")
        print(correlated_columns)
        
        # Find uncorrelated columns (columns to keep from original data)
        uncorrelated_columns = [col for col in self.data.columns if col not in correlated_columns]
        print(f"Keeping {len(uncorrelated_columns)} uncorrelated variables:")
        print(uncorrelated_columns)
        
        # Select subset of data with correlated variables
        correlated_data = self.data[correlated_columns]
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(correlated_data)
        
        # First fit PCA with all components to calculate explained variance
        if variance_threshold is not None:
            if not (0 < variance_threshold <= 1.0):
                raise ValueError("variance_threshold must be between 0 and 1")
                
            temp_pca = PCA()
            temp_pca.fit(scaled_data)
            
            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            
            # Find how many components are needed to reach the variance threshold
            if max(cumulative_variance) < variance_threshold:
                print(f"Warning: Even with all components, only {max(cumulative_variance)*100:.1f}% " 
                      f"variance is explained. This is less than the requested {variance_threshold*100:.1f}%.")
                n_components_for_threshold = len(cumulative_variance)
            else:
                n_components_for_threshold = np.argmax(cumulative_variance >= variance_threshold) + 1
            
            print(f"Using {n_components_for_threshold} components to explain {variance_threshold*100:.1f}% of variance")
            n_components = n_components_for_threshold
        
        # Initialize PCA with determined number of components
        self.pca_correlated = PCA(n_components=n_components)
        
        # Fit and transform the data
        self.transformed_correlated = self.pca_correlated.fit_transform(scaled_data)
        
        # Store explained variance
        self.explained_variance_correlated = self.pca_correlated.explained_variance_ratio_
        
        # Create a DataFrame from the transformed data with PCA component columns
        cols = [f'PC{i+1}_corr' for i in range(self.transformed_correlated.shape[1])]
        transformed_df = pd.DataFrame(data=self.transformed_correlated, columns=cols, index=correlated_data.index)
        
        # Store the correlated columns for reference
        self.correlated_columns = correlated_columns
        
        # Display the results
        self._display_pca_results(self.pca_correlated, "Correlated Variables")
        
        # Create a new dataframe with uncorrelated original features + PCA components
        result_df = pd.DataFrame(index=self.data.index)
        
        # Add uncorrelated original features
        for col in uncorrelated_columns:
            result_df[col] = self.data[col]
            
        # Add PCA components
        for col in transformed_df.columns:
            result_df[col] = transformed_df[col]
        
        # Store both dataframes for reference
        self.pca_components_df = transformed_df
        self.result_df = result_df
        
        return result_df
    
    def _display_pca_results(self, pca_model, title):
        """
        Display PCA results including explained variance and cumulative variance.
        
        Parameters:
        -----------
        pca_model : sklearn.decomposition.PCA
            The fitted PCA model to display results for.
        title : str
            Title for the plot.
        """
        # Print explained variance
        print(f"\n--- PCA Results for {title} ---")
        print("Explained variance ratio by component:")
        for i, var in enumerate(pca_model.explained_variance_ratio_):
            print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        
        # Print cumulative explained variance
        cumulative = np.cumsum(pca_model.explained_variance_ratio_)
        print("\nCumulative explained variance:")
        for i, cum_var in enumerate(cumulative):
            print(f"PC1-PC{i+1}: {cum_var:.4f} ({cum_var*100:.2f}%)")
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(pca_model.explained_variance_ratio_) + 1), 
                pca_model.explained_variance_ratio_, alpha=0.5, 
                label='Individual explained variance')
        plt.step(range(1, len(cumulative) + 1), cumulative, where='mid',
                 label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.title(f'Explained variance by principal components ({title})')
        plt.xticks(range(1, len(pca_model.explained_variance_ratio_) + 1))
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self):
        """
        Plot a heatmap of the correlation matrix.
        """
        if self.correlation_matrix is None:
            self.correlation_matrix = self.data.corr().abs()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', fmt='.2f', 
                   linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()
        plt.show()
    
    def plot_pca_components(self, pca_type='all', components=[0, 1]):
        """
        Plot a scatter plot of data points in the principal component space.
        
        Parameters:
        -----------
        pca_type : str, default='all'
            Type of PCA to plot. Options are 'all' or 'correlated'.
        components : list, default=[0, 1]
            The components to plot (zero-indexed).
        """
        if pca_type == 'all' and self.transformed_all is not None:
            transformed = self.transformed_all
            pca_model = self.pca_all
            title = "All Variables"
        elif pca_type == 'correlated' and self.transformed_correlated is not None:
            transformed = self.transformed_correlated
            pca_model = self.pca_correlated
            title = "Correlated Variables"
        else:
            print(f"PCA of type '{pca_type}' has not been performed yet.")
            return
        
        # Check if requested components are valid
        if max(components) >= transformed.shape[1]:
            print(f"Invalid components. Max component index is {transformed.shape[1]-1}")
            return
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(transformed[:, components[0]], transformed[:, components[1]], 
                   alpha=0.7, edgecolors='w', s=50)
        
        # Add labels and title
        plt.xlabel(f'Principal Component {components[0]+1} '
                  f'({pca_model.explained_variance_ratio_[components[0]]:.2%})')
        plt.ylabel(f'Principal Component {components[1]+1} '
                  f'({pca_model.explained_variance_ratio_[components[1]]:.2%})')
        plt.title(f'PCA: {title} - PC{components[0]+1} vs PC{components[1]+1}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate a sample dataset
    # Replace this with your actual data loading code
    # For example: df = pd.read_csv("your_data.csv")
    np.random.seed(42)
    
    # Create a dataset with some correlated variables
    n_samples = 100
    
    # Independent variables
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    
    # Correlated variables
    x3 = x1 * 0.8 + np.random.normal(0, 0.2, n_samples)  # Correlated with x1
    x4 = x1 * 0.9 + np.random.normal(0, 0.1, n_samples)  # Strongly correlated with x1
    x5 = x2 * 0.7 + np.random.normal(0, 0.3, n_samples)  # Correlated with x2
    x6 = np.random.normal(0, 1, n_samples)  # Independent
    
    # Create dataframe
    df = pd.DataFrame({
        'var1': x1,
        'var2': x2,
        'var3': x3,
        'var4': x4,
        'var5': x5,
        'var6': x6
    })
    
    # Initialize PCA analyzer
    pca_analyzer = PCAAnalyzer(df)
    
    # Plot correlation heatmap
    pca_analyzer.plot_correlation_heatmap()
    
    # Perform PCA on all variables
    print("\n=== PCA on all variables ===")
    transformed_all = pca_analyzer.perform_pca_all()
    
    # Perform PCA on correlated variables
    print("\n=== PCA on correlated variables (correlation > 0.7) ===")
    transformed_corr = pca_analyzer.perform_pca_correlated(correlation_threshold=0.7)
    
    # Plot PCA components
    if transformed_all is not None:
        pca_analyzer.plot_pca_components(pca_type='all')
    
    if transformed_corr is not None:
        pca_analyzer.plot_pca_components(pca_type='correlated')