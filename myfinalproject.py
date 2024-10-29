

from google.colab import drive
drive.mount('/content/drive')

# Importing Libraries
import pandas as pd # for data analysis
import numpy as np # for working with arrays
import matplotlib.pyplot as plt # For data visualization
plt.rcParams.update({'font.size':20})
import seaborn as sns # For data visualization
import random # For random data point generation
import re #use Regular Expression Module to help me clean the data
from sklearn.model_selection import StratifiedKFold #to split the data in a stratified manner


from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors

# Loading the dataset into a pandas dataframe
df_data = pd.read_csv("/content/drive/MyDrive/final_project/nutrition.csv", index_col= 0)

"""# Data Cleaning"""

#Fill the NaN value with 0
df_data.fillna(0, inplace = True)

#Looping in each non-numerical features except the name feature. 
#It will remove all the non-numerical text in the datafirst.
#Then, it willtransform all the numerical data except calories to have the same measurement (gram). 
for col in df_data.drop('name',axis = 1).select_dtypes(exclude = 'number').columns:
    for i in df_data[col]:
        if i == '0' or i == 0:
            pass
        else:
            point = re.findall('[a-zA-Z]+',i)[0]
            replace = []
            if point == 'mg':
                for j in df_data[col]:
                    if j == '0' or j == 0:
                        replace.append(float(j))
                    else:
                        replace.append(float(re.sub('[a-zA-Z]','',j))/1000)
            elif point == 'mcg':
                for j in df_data[col]:
                    if j == '0' or j == 0:
                        replace.append(float(j))
                    else:
                        replace.append(float(re.sub('[a-zA-Z]','',j))/1000000)  
            else:
                 for j in df_data[col]:
                    if j == '0' or j == 0:
                        replace.append(float(j))
                    else:       
                        replace.append(float(re.sub('[a-zA-Z]','',j)))
                        
            df_data[col] = replace    
            df_data.rename({col:col+'(g)'}, axis =1, inplace = True)
            break
df_data.head(3)

#using .drop function to drop few columns that seems unnncessary.
df_data2 = df_data.drop(columns=['serving_size(g)','lucopene','theobromine(g)','alcohol(g)','folic_acid(g)','carotene_alpha(g)', 'carotene_beta(g)',
       'cryptoxanthin_beta(g)', 'lutein_zeaxanthin(g)','tocopherol_alpha(g)','copper(g)', 'alanine(g)',
       'arginine(g)', 'aspartic_acid(g)', 'cystine(g)', 'glutamic_acid(g)', 'caffeine(g)',
       'glycine(g)', 'histidine(g)', 'hydroxyproline(g)', 'isoleucine(g)',
       'leucine(g)', 'lysine(g)', 'methionine(g)', 'phenylalanine(g)',
       'proline(g)', 'serine(g)', 'threonine(g)', 'tryptophan(g)',
       'tyrosine(g)', 'valine(g)','total_fat(g)','sodium(g)','choline(g)',
       'fructose(g)', 'galactose(g)', 'glucose(g)', 'lactose(g)', 'maltose(g)',
       'vitamin_b6(g)','vitamin_c(g)','vitamin_d(g)','vitamin_e(g)','calcium(g)','irom(g)',
       'magnesium(g)','manganese(g)','fat(g)',
       'sucrose(g)', 'saturated_fatty_acids(g)','saturated_fat(g)','niacin(g)','pantothenic_acid(g)',
       'thiamin(g)','phosphorous(g)','potassium(g)','selenium(g)','zink(g)','sugars(g)','water(g)',
       'fatty_acids_total_trans(g)', 'ash(g)',	'folate(g)','riboflavin(g)','vitamin_a_rae(g)','vitamin_k(g)','vitamin_b12(g)','vitamin_a(g)'], axis=1)
df_data2

#Adding new feature called healthy fat which is the sum of monounsaturated_fatty and polyunsaturated_fatty
df_data2['healthy fat'] = df_data2['monounsaturated_fatty_acids(g)'] + df_data2['polyunsaturated_fatty_acids(g)']

df_data2 = df_data2.drop(columns=['monounsaturated_fatty_acids(g)','polyunsaturated_fatty_acids(g)'])
df_data2

#heatmap to see the relationship between features
plt.figure(figsize = (10,10))
coor_matirx = sns.heatmap(df_data2.corr(), annot=True) #coorelation between columnd
plt.show()

#Transform features by scaling each feature to a given range
X= df_data2.iloc[:,1:]
sc = MinMaxScaler() #scale and translate each feature individually to range zero and one
X_transormed = sc.fit_transform(X)

#Elbow Criterion to find the best K
error = {}
for k in range(1, 20):
  kmeans = KMeans(n_clusters=k,init='random', random_state=None).fit(X_transormed)
  cluster_idx = kmeans.labels_ #Labels of each point
  error[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure(figsize=(10,5))
plt.plot(list(error.keys()), list(error.values()))
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.xticks(np.arange(0, (k)+1, 1))

plt.show()

"""#K-means using sklearn"""

#k-means function
kmeans = KMeans(n_clusters=5,init='random',n_init=10, random_state=None).fit(X_transormed)

#sum of squared distance between each point and the centroid in a cluster.
print('error',kmeans.inertia_)

cluster_idx = kmeans.labels_ #Labels of each point
values,count = np.unique(cluster_idx, return_counts=True) #how many data point is in each cluster
print('clusters',count)

#using TSNE to visualise the structure of high dimensional data in 2 dimensions
#X_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(X_transormed)

#Visualizing
plt.figure(figsize=(10,10)) 
sns.scatterplot(data=df_data2, x='carbohydrate(g)', y= 'protein(g)',palette ='rocket', hue = cluster_idx)
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
sns.scatterplot(data=df_data2, x='healthy fat', y= 'protein(g)',palette ='rocket', hue = cluster_idx)
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
sns.scatterplot(data=df_data2, x='fiber(g)', y= 'protein(g)',palette ='rocket', hue = cluster_idx)
plt.legend()
plt.show()

import plotly.express as px
fig = px.scatter_3d(df_data2, x='carbohydrate(g)', y='protein(g)', z='healthy fat',
              color=cluster_idx, hover_data=['name'])
fig.update_traces(marker_size=4)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

"""#K_means using my algorithm"""

# Euclidean distance of two vectors
def dist(x1,x2):
  '''
  This function takes two vectors and computes the euclidean distance between them
  '''
  return np.sqrt(np.sum((x1-x2)**2, axis=0))

def assignment(k,X,centroids):
  '''
  This function is the assigment step of kmeans algorithm
  Inputs: 
  k: number of clusters
  X: The data points
  centroids: The initial or former step's centroids
  Output:
  cluster_idx : The cluster id assigned to data points
  '''
  # initializing a matrix of zeros for distances of each data point to each centroid
  # with the shape of (number of data points, number of clusters)
  dist_to_centers = np.zeros((X.shape[0],k))
  # Looping over data points
  for j in range(X.shape[0]):
    # Looping over number of clusters
    for i in range(k):
      # Using the dist function two compute the euclidean distance between each data point and centroid
      dist_to_centers[j,i] = dist(X[j],centroids[i])
  # Finding the cluster idx by taking argmin of the distances 
  cluster_idx = np.argmin(dist_to_centers,axis=1)
  return cluster_idx

def update(k, cluster_idx,X):
  '''
  This function updates the centriods 

  Inputs:
  k : number of clusters
  cluster_idx : cluster indexes
  X : data points
  Outputs:
  centroids: newly computed centroids 
  '''
  # Initializing centroids with an empty python list
  centroids = []

  # Looping over number of clusters
  for i in range(k):
    # taking the mean of data points which are from the same cluster and appending it to the list
    centroids.append(np.mean(X[cluster_idx==i],axis=0))
  return centroids

def distances_to_centroids(input, clusters, centroids):
  '''
  This function computes the Sum of squared distances of data points to their closest cluster center (wcss)
  Inputs: 
  input: data points
  clusters: final clusters
  centroids: final centroids
  Outputs:
  sum square error of distances (wcss)
  '''
  # setting the sum to zero
  sum_distances = 0
  # Finding number of clusters
  num_clusters = max(clusters) +1 
  # Looping over number of clusters
  for k in range(num_clusters):
    # Selecting the data related to each cluster
    data = input[clusters == k]
    # Selecting the centeroid related to each cluster
    center = centroids[k]
    # Looping over the data points of each cluster
    for j in range(data.shape[0]):
      # Adding square of distances to the final wcss
      sum_distances += dist(data[j],center)**2
  return sum_distances

def make_initial_centriods(input, k):
  '''
  A function to generate initial centroids
  Inputs:
  input: data points
  k = number of clusters
  Outputs: 
  initial centroids : the initial chosen centroids from the input data points
  '''
  # Random ndex of centroids with size of k
  cent_idx = random.sample(range(input.shape[0]),k)
  initial_centroids = []
  # Appending the selected initial centroids to the list
  for i in cent_idx:
      initial_centroids.append(input[i])
  return initial_centroids

def k_means(input,k, r =10):
  '''
  Main kmeans function that perform assigment and update steps, as well as r different inits
  and computes the wcss errors and choose the best one
  Inputs:
  input : data points
  k : number of clusters
  r : number of inits
  Outputs:
  all_iter_clusters[best_iter][-1]: The best cluster indexes from all the inits
  all_iter_centriods[best_iter][-1]: The best cluster centroids from all the inits
  errors[best_iter]: The lowest wcss error of best init
  best_iter:  The index of best iteration (r)
  all_iter_clusters: All the clusters from all the inits and from inside each init to help with plots
  all_iter_centriods : All the centroids from all the inits and from inside each init to help with plots
  '''

  # All the clusters and centroids from each init and from each step
  all_iter_clusters = []
  all_iter_centriods = []
  errors = []
  # Looping over number of inits (r)
  for iter in range(r):
    # Making initial centroids
    init_centroids = make_initial_centriods(input, k)

    # all clusters inside each init at each step
    all_clusters = []
    initial_clusters = np.zeros((input.shape[0],))
    all_clusters.append(initial_clusters)
    # Loss is the difference of new cluster indexes and the previous one
    # If zero means there is no change to cluster indexes and we stop it
    loss = 1
    i = 0
    all_centroids = []
    all_centroids.append(init_centroids)

    # Continue to do assigment and update steps until the loss converges to zero
    # If zero means there is no change to cluster indexes and we stop it
    while loss != 0: 

      # Assignment step
      clusters = assignment(k,input,all_centroids[i])
      # Update step
      new_centroids = update(k, clusters,input)
      all_clusters.append(clusters)
      all_centroids.append(new_centroids)
      # Loss of each iteration, difference between cluster indexes and previous step cluster indexes
      loss = np.sum(clusters != all_clusters[i])
      i+=1

    # WCSS error at the end of each init (r)
    error = distances_to_centroids(input, all_clusters[-1], all_centroids[-1])
    errors.append(error)


    all_iter_clusters.append(all_clusters)
    all_iter_centriods.append(all_centroids)

  # Best init error index
  best_iter = np.argmin(errors)
  print('--------------')
  print(f'best sum square error is: {errors[best_iter]}')
  print('--------------')
  return all_iter_clusters[best_iter][-1],all_iter_centriods[best_iter][-1],errors[best_iter],best_iter,  all_iter_clusters, all_iter_centriods

# Running our kmeans for k =5 on our dataset with 10 different random inits (r)
best_cluster, best_centroid, best_error,best_iter, clusters, centroids = k_means(X_transormed, k=5, r= 10)

values,count = np.unique(best_cluster, return_counts=True) #how many data point is in each cluster

print('clusters',count)
print('--------------')

#visulizing
plt.figure(figsize=(10,10)) 
sns.scatterplot(data=df_data2, x='carbohydrate(g)', y= 'protein(g)',palette ='rocket', hue = best_cluster)
plt.legend()
plt.show()

"""#supervised Learning(KNN)

###Load and cleaning the data
"""

df_data3 = df_data2.drop(columns=['name'], axis=1)
data_names = df_data2['name']

train, test = train_test_split(df_data3, test_size=0.2) #split the data set for train and test set

y_train = train.iloc[:,0]
x_train = train.iloc[:,1:]
y_test = test.iloc[:,0]
x_test = test.iloc[:,1:]

scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

score_training = [] 
score_test = [] #to store error values for different k

for K in range(1,30):
    neigh = KNeighborsRegressor(n_neighbors = K, weights='distance')
    neigh.fit(x_train_scaled, y_train)  #fit the model

    train_score = neigh.score(x_train_scaled, y_train) 
    test_score = neigh.score(x_test_scaled, y_test) 

    score_training.append(train_score) #store error values
    score_test.append(test_score) #store error values


plt.figure(figsize= (10, 6))
plt.plot(np.arange(1,30), score_training, c='r', label= 'Training')
plt.plot(np.arange(1,30), score_test, c='b', label = 'Testing')
plt.xlabel('K')
plt.ylabel('score (%)')
plt.legend()
plt.show()

neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(x_train_scaled, y_train)

print(neigh.score(x_train_scaled, y_train))
print(neigh.score(x_test_scaled, y_test))

"""# My KNN"""

# Euclidean distance of two vectors
def dist(x1,x2):
  '''
  This function takes two vectors and computes the euclidean distance between them
  '''
  return np.sqrt(np.sum((x1-x2)**2, axis=0))

def find_neighbors(X_train, X_test, k):
  """
  This functions find the neighbors of a test data point
  Inputs:
  X_train : The training dataset
  X_test : The test data point
  k: The number of neighbors
  Outputs: 
  neighbors_idx: The neighbor indexes 
  neighbors_distances: The neighbor distances from the test data point
  """
  distances = np.zeros((X_train.shape[0],1))
  neighbors = []
  for i in range(X_train.shape[0]):
    distances[i] = dist(X_train[i], X_test)
  
  sorted_distances = np.sort(distances, axis=0)
  sorted_indexes = np.argsort(distances, axis=0)
  neighbors_distances = sorted_distances[:k]
  neighbors_idx = sorted_indexes[:k]
  return neighbors_idx, neighbors_distances

def predict(y_train, neighbors_idx):
  target_values = []
  for i in range(neighbors_idx.shape[0]):
    target_values.append(y_train[neighbors_idx])
  return np.mean(target_values)


def knn(X_train, y_train, X_test, k):
  neighbors_idx, neighbors_distances = find_neighbors(X_train, X_test, k)
  y_pred = predict(y_train, neighbors_idx)
  return y_pred, neighbors_idx, neighbors_distances

X_train = X.values[:, 1:]
y_train = X.values[:, 0]


X_test= X.values[486, 1:]
y_test= X.values[486, 0]
print('my test food', df_data.loc[486, 'name'] )

y_pred, neighbors, distances = knn(X_train, y_train, X_test, k=6)

for i in range(len(neighbors)):
  print(df_data.loc[neighbors[i],'name'].values)

print(y_pred)
print(y_test)