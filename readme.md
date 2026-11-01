# Customer Segmentation using K-Means Clustering

A Data Science and Machine Learning project implementing K-Means Clustering to group retail store customers based on purchase behaviors and financial demographics.

## Overview

Customer segmentation helps retail businesses understand consumer behavior, enabling targeted marketing campaigns, personalized recommendations, and enhanced customer engagement. This project applies Unsupervised Machine Learning via the K-Means clustering algorithm on customer annual income and spending performance data.

## Features

* **Data Preprocessing & Feature Selection:** Cleans and selects feature attributes (`Annual Income` and `Spending Score`).
* **Elbow Method & Silhouette Analysis:** Evaluates optimal cluster parameters ($K$).
* **K-Means Model Execution:** Segments customers into distinct behavioral profiles using `scikit-learn`.
* **Cluster Profiling:** Generates statistical summaries for each customer group.

## Dataset

The analysis uses the **Mall Customers Dataset** (`Mall_Customers.csv`), containing:
* `CustomerID`: Unique customer identifier.
* `Gender`: Gender of the customer.
* `Age`: Age of the customer.
* `Annual Income (k$)`: Annual income of the customer in thousands of dollars.
* `Spending Score (1-100)`: Score assigned by the store based on customer behavior and spending nature.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Customer-Segmentation-KMeans.git
   cd Customer-Segmentation-KMeans
   ```

2. Install the required Python packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

## Usage

Run the clustering script with default settings:
```bash
python customer_segmentation.py --data Mall_Customers.csv
```

Specify a custom number of clusters ($K$):
```bash
python customer_segmentation.py --data Mall_Customers.csv --clusters 5
```

## License

MIT
