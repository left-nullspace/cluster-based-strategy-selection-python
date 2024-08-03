# cluster-based-strategy-selection
Exploring the clustering of pre-designed trading strategies to see if there is any potential for creating uncorrelated investment portfolios.


---

#### Description
This project is designed to address the question, "Which of my trading strategies should I choose?" By employing clustering techniques, we group various trading strategies based on a comprehensive set of features. From these groups, we identify and select the "best" strategy within each cluster. The selected strategies are then combined to evaluate their collective performance and analyze the correlation between them. This approach not only helps in selecting optimal strategies but also in understanding the dynamics between different trading methods.

#### Features
- **Data Loading:** Automatically imports and cleans trading data and ETF returns.
- **Feature Engineering:** Calculates a comprehensive set of financial metrics.
- **Dimensionality Reduction:** Applies Principal Component Analysis (PCA) to reduce the number of features.
- **Clustering:** Implements the K-Means algorithm with an Elbow method to determine the optimal number of clusters.
- **Performance Visualization:** Plots equity curves for individual and combined strategies and provides a comparison across different metrics.
- **Correlation Analysis:** Includes functionality to analyze and visualize the correlation among selected strategies and across clusters.

#### Usage
Run the main.py file

#### Files and Modules
- `data_loading.py`: Handles data fetching and preprocessing.
- `feature_engineering.py`: Contains functions for computing financial metrics and applying PCA.
- `clustering.py`: Includes methods for clustering data and evaluating the clustering performance.
- `portfolio.py`: Defines methods for analyzing the performance of strategies within clusters.
- `main.py`: The main script that runs the entire analysis pipeline.

