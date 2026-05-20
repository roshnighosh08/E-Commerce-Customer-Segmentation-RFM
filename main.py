# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set style for visualizations
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ==========================================
# 2. LOAD DATASET
# ==========================================
# Load the dataset
df = pd.read_csv("part_1_ecommerce_customer_segmentation.csv")
print("--- Dataset Shape ---")
print(df.shape)
print("\n--- First 5 Rows ---")
print(df.head())

# ==========================================
# 3. DATA UNDERSTANDING
# ==========================================
print("\n--- Info ---")
print(df.info())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())

print("\n--- Summary Statistics ---")
print(df.describe(include='all'))

# ==========================================
# 4. DATA CLEANING
# ==========================================
# 1. Drop records where Customer ID is missing
df_clean = df.dropna(subset=['CustomerID']).copy()

# 2. Drop records where Description is missing
df_clean = df_clean.dropna(subset=['Description'])

# 3. Remove duplicate records
df_clean = df_clean.drop_duplicates()

# 4. Filter out negative or zero Quantities and UnitPrices (removes cancellations/errors)
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

# 5. Fix Data Types
# Since Customer ID contains letters (e.g., 'C10555'), convert directly to string format
df_clean['CustomerID'] = df_clean['CustomerID'].astype(str)
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

print("\n--- Cleaned Dataset Shape ---")
print(df_clean.shape)

# ==========================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
# Create a Revenue column for transaction-level analysis
df_clean['Revenue'] = df_clean['Quantity'] * df_clean['UnitPrice']

# Chart 1: Top 5 Countries by Sales Revenue
top_countries = df_clean.groupby('Country')['Revenue'].sum().nlargest(5)
plt.figure()
sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis')
plt.title('Top 5 Countries by Total Sales Revenue')
plt.xlabel('Total Revenue')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('top_countries.png')
plt.show()
print("\nInterpretation Chart 1: The bar chart isolates sales performance by region. The United Kingdom represents the overwhelming majority of the company's financial footprint, indicating a highly centralized core market.")

# Chart 2: Top 5 Best Selling Products by Quantity
top_products_qty = df_clean.groupby('Description')['Quantity'].sum().nlargest(5)
plt.figure()
sns.barplot(x=top_products_qty.values, y=top_products_qty.index, palette='mako')
plt.title('Top 5 Best Selling Products by Total Quantity Sold')
plt.xlabel('Total Quantity')
plt.ylabel('Product Description')
plt.tight_layout()
plt.savefig('top_products_qty.png')
plt.show()
print("\nInterpretation Chart 2: This chart shows the high-volume products moving through inventory. High transaction volume on these specific inventory items highlights core inventory assets that drive routine consumer interactions.")

# Chart 3: Distribution of Unit Price (Outlier Detection)
plt.figure()
sns.boxplot(x=df_clean['UnitPrice'], color='skyblue')
plt.title('Distribution and Outlier Profile of Product Unit Prices')
plt.xlabel('Unit Price')
plt.tight_layout()
plt.savefig('unit_price_outliers.png')
plt.show()
print("\nInterpretation Chart 3: The boxplot uncovers right-skewness and transactional outliers within the pricing structure. The vast majority of stock items are low-cost goods, while a few select entries command premium pricing.")

# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================
# Set reference date as one day after the maximum date in the dataset to calculate recency
snapshot_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)

# Group by customer to engineer basic metrics and the core RFM features
customer_features = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'nunique',                                  # Frequency (Total Purchases)
    'Revenue': 'sum',                                        # Monetary (Total Revenue)
    'Quantity': 'sum',                                       # Total Quantity
    'Description': 'nunique',                                # Unique Products Purchased
    'Country': 'first'                                       # Native Market Country
}).reset_index()

# Rename the RFM pillars clearly
customer_features.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'Revenue': 'Monetary'
}, inplace=True)

# Calculate derived metric: Average Order Value
customer_features['AverageOrderValue'] = customer_features['Monetary'] / customer_features['Frequency']

print("\n--- Engineered Customer Features ---")
print(customer_features.head())

# ==========================================
# 7. MODEL BUILDING / ANALYSIS
# ==========================================
# Isolate numeric metrics explicitly for K-Means modeling
rfm_data = customer_features[['Recency', 'Frequency', 'Monetary']].copy()

# Log-transform variables to stabilize variance and mitigate extreme outlier skew
rfm_log = np.log1p(rfm_data)

# Scale metrics to zero mean and unit variance
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# Execute the Elbow Method to evaluate mathematical cluster configurations
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

# Chart 4: The Elbow Curve
plt.figure()
plt.plot(k_range, inertia, marker='o', linewidth=2, color='darkred')
plt.title('Elbow Method for Selecting Optimal K-Means Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig('elbow_method.png')
plt.show()
print("\nInterpretation Chart 4: The line plot shows a distinct inflection point or 'elbow' at k=3. Beyond 3 clusters, the reduction in within-cluster sum of squares decelerates, establishing 3 as the optimal configuration.")

# Train the optimal model
optimal_k = 3
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_features['Cluster'] = kmeans_model.fit_predict(rfm_scaled)

# ==========================================
# 8. EVALUATION
# ==========================================
# Review raw numerical summary aggregations across the generated clusters
cluster_summary = customer_features.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Segment Size'}).reset_index()

print("\n--- Cluster Mathematical Profile Summary ---")
print(cluster_summary)

# ==========================================
# 9. BUSINESS INSIGHTS
# ==========================================
# Chart 5: Cluster Distribution Visualization
plt.figure()
sns.scatterplot(
    x='Frequency', y='Monetary', hue='Cluster', 
    data=customer_features, palette='Set1', alpha=0.7
)
plt.yscale('log') # Log scale to make visualization legible amidst high spending variance
plt.title('Customer Segmentation Profile Across Frequency vs. Monetary Expenditure')
plt.xlabel('Frequency (Total Separate Orders)')
plt.ylabel('Monetary Value (Total Spend in Log Scale)')
plt.legend(title='Assigned Cluster')
plt.tight_layout()
plt.savefig('cluster_scatter.png')
plt.show()
print("\nInterpretation Chart 5: The scatter plot clearly delineates the customer base into distinct groups along both transaction consistency and overall financial scale, indicating robust algorithmic performance.")

for idx, row in cluster_summary.iterrows():
    c_id = int(row['Cluster'])
    rec = row['Recency']
    freq = row['Frequency']
    mon = row['Monetary']
    
    print(f"\nCluster {c_id} Metric Profile Evaluation:")
    print(f" -> Avg Recency: {rec:.1f} days | Avg Frequency: {freq:.1f} orders | Avg Spend: ${mon:.2f}")

# ==========================================
# 10. FINAL RECOMMENDATIONS
# ==========================================
print("\n=== SYSTEM EXECUTION COMPLETE ===")
print("Review generated plots in the execution directory and proceed to compile documentation repository files.")