import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Clustering:
    @staticmethod
    def elbow_method(data,start=3, end=20, n_init=150):
        #z score normalization
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        #calculate intertia
        inertia_values = [] #intiialize empty list

        #loop from start to end for collecting intertia values for elbow method
        for n in range(start, end+1):
            kmeans = KMeans(n_clusters=n, n_init=n_init, random_state=7)
            kmeans.fit(scaled_data)
            inertia_values.append(kmeans.inertia_)
        plt.figure(figsize=(10, 6))
        plt.plot(range(start, end + 1), inertia_values, marker='o')
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.show()


    @staticmethod
    def cluster_data(data, n_clusters=10, n_init=150):
        # uses Lloyd's algorithm (default in KMeans)
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=7)
        clusters = kmeans.fit_predict(data)
        
        #calculate and print the common evaluation metrics
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(data, clusters)
        calinski_harabasz = calinski_harabasz_score(data, clusters)
        
        print(f"Inertia: {inertia}")
        print(f"Silhouette Score: {silhouette_avg}")
        print(f"Calinski-Harabasz Index: {calinski_harabasz}")

        # Create a DataFrame with cluster labels, aligning with the input data's index
        cluster_df = pd.DataFrame(clusters, index=data.index, columns=['Cluster']) # index is strats
        print(f"cluster_df: {cluster_df}")
        return cluster_df