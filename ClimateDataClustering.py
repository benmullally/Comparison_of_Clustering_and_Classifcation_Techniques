## Code for clustering task, including preprocessing and all models fitted to the climate dataset

#primary libraries used
import numpy as np
import time #records time 
import matplotlib.pyplot as plt
import pandas as pd #for ease of handling dataframes
import seaborn as sns #prettier plots

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

###########################################
###########################################
#pre-processing


#import of climate data set through pandas
#reads file and converts to pandas dataframe
#pandas can freely swap between dataframes and numpy arrays, both used
climData = pd.read_csv('Data Files/ClimateDataBasel.csv',header=None)
#get columns and rows
print(climData.shape)

#column names to assign to dataset
cols = ["TempMin", "TempMax", "TempMean", "HumidityMin", "HumidityMax", "HumidityMean", "SeaLevelPressureMin", "SeaLevelPressureMax", 
                    "SeaLevelPressureMean", "TotalPrecipitation", "SnowfallAmount", "SunshineDuration", "WindGustMin", "WindGustMax", "WindGustMean",
                    "WindSpeedMin", "WindSpeedMax", "WindSpeedMean"]
climData.columns = cols

#1) Feature Selection

#search for any rows with NA, none found -> no need to remove rows 
print(climData[climData.isna().any(axis=1)])

# histogram to show distributions of features
plt.figure()
sns.histplot([climData['TempMean'],climData['HumidityMean'],climData['WindSpeedMean']], bins=100)
plt.show()

plt.figure()
sns.histplot([climData['SeaLevelPressureMean']], bins=100)
plt.show()

#sunshine not gaussian so do not do outlier test on it
plt.figure() 
sns.histplot(climData["SunshineDuration"], bins=100)
plt.show() 

#show correlation of features
corrmat = pd.DataFrame(climData).corr()
plt.figure(figsize=(6,4))
sns.set(font_scale=1)
sns.heatmap(corrmat,cmap="coolwarm")
plt.title("Correlation of features")
plt.show()


#1407 of 1763 (80%) rows have less than 2mm of rainfall. This feature is unlikely to show any trend so remove
print(len(climData[climData["TotalPrecipitation"] <2]))
#93% data zero for snowfall too 
print(len(climData[climData["SnowfallAmount"] == 0]))
#drop snowfall column
climData = climData.drop(["SnowfallAmount","TotalPrecipitation"],axis=1)
#must rename columns
cols = ["TempMin", "TempMax", "TempMean", "HumidityMin", "HumidityMax", "HumidityMean", "SeaLevelPressureMin", "SeaLevelPressureMax", 
                    "SeaLevelPressureMean", "SunshineDuration", "WindGustMin", "WindGustMax", "WindGustMean",
                    "WindSpeedMin", "WindSpeedMax", "WindSpeedMean"]
#2) Standardize
#look for rows with outliers using standardisation
#sklearn has function StandardScaler() that automatically performs standardisation of each feature
# the process is (x-u)/s where x is a feature, u is the mean and s is the standard deviation
stscaler = StandardScaler()
climData = pd.DataFrame(stscaler.fit_transform(climData), columns=cols)

#investigate extreme values
#3) Detect global outliers (if <-3 or >3);
#done for wind, humidity and sea level due to gaussian nature.
outlierRows = climData[((climData["WindSpeedMean"] > 3) | (climData["WindSpeedMean"]<-3))].index
outlierRows = [int(x) for x in outlierRows.tolist()]

#remove rows 
climData = climData.drop(outlierRows, axis=0)

outlierRows = climData[((climData["HumidityMean"] > 3) | (climData["HumidityMean"]<-3))].index
outlierRows = [int(x) for x in outlierRows.tolist()]
#remove rows
climData = climData.drop(outlierRows, axis=0)

outlierRows = climData[((climData["SeaLevelPressureMean"] > 3) | (climData["HumidityMean"]<-3))].index
outlierRows = [int(x) for x in outlierRows.tolist()]
print(outlierRows)
#remove rows
climData = climData.drop(outlierRows, axis=0)
# use climDataScaled for rest of analysis
climDataScaled = climData

# do not need to look at temperature as no outliers

#4) Normalise to convert [-3;3] n → [0;1]n
#destandardise to get original data i case needed later in analysis
climData = pd.DataFrame(stscaler.inverse_transform(climData), columns=cols)


#construct pca using 8 components due to high complexity 
pca = PCA(n_components=8)
pca.fit(climDataScaled)

transformedData = pca.transform(climDataScaled)

#plot data in 2d
plt.figure(figsize=(6,4))
plt.plot(transformedData[:,0],transformedData[:,1],".")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")

#plot variance explained of pca
plt.figure(figsize=(6,4))
plt.bar([1, 2, 3, 4,5,6,7,8], pca.explained_variance_ratio_,tick_label=[1, 2, 3, 4, 5,6,7,8])
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.show()

#plot cumulative variance explained to find best number of components
plt.figure(figsize=(6,4))
plt.plot(range(1,9,1), pca.explained_variance_ratio_.cumsum(),marker="o",linestyle="--")
plt.xlabel("Principal Component")
plt.ylabel("Cum Variance Explained (%)")
plt.show()

#3d plot no longer used
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(transformedData[:,0],transformedData[:,1],transformedData[:,2],s=0.1)
# # fig.show()

#making pca loadings and visualising biplot
# credit to https://www.reneshbedre.com/blog/principal-component-analysis.html - help on code
loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = climData.columns.values
loadings_df = loadings_df.set_index('variable')

#plot correlation between original features and principal commponents
plt.figure()
sns.heatmap(loadings_df, annot=True, cmap="coolwarm")
plt.title("Correlation of Features to PCA")
plt.show()

#show pca loadings
fig, ax = plt.subplots()
ax.scatter(x=loadings[0], y=loadings[1], c="black")
#pick hgand selected values to show groups of features
for i in [2,5,8,9,15]:
    #annotate values
    ax.annotate(climData.columns.values[i], (loadings[0][i]+0.03, loadings[1][i]+0.01), fontsize=12)
    
#plot lines from origin
for i in range(len(loadings_df)):
    ax.arrow(0,0,loadings[0][i], loadings[1][i], color='black', alpha=0.3)
ax.set_title("Loadings Plot of Features")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")    
plt.show()

#propose to take just first 4 components -> explain almost 90% of variance
df = transformedData[:,0:4]
#dataframe used for graphing
graphKMdata = pd.DataFrame(df)
#dataframe used for model building
newPCAdata = pd.DataFrame(df)


#same dataset but keeps later steps simpler


#model function that runs model inputted for different k and then plots the score graphs
def model(mod, newPCAdata, name, affinity=None, threshold=None):
    distortions = []
    silhouetteVals = []
    ch = {}
    db = {}
    #up to k=15
    for i in range(2,15,1):
        #creates relevant sklearn model
        #thrshold needed for BIRCH
        if threshold is not None:
            km = mod(n_clusters=i, threshold=threshold)
        #any model
        elif affinity is None:
            km = mod(n_clusters=i, random_state=1)
        ##Spectral model with affinity
        else:
            km = mod(n_clusters=i, random_state=1, affinity=affinity)
        labels = km.fit_predict(newPCAdata)
        #distortions only available for some models
        if affinity is None and threshold is None:
            distortions.append(km.inertia_)
        silhouetteVals.append(silhouette_score(newPCAdata, km.labels_))
        ch_index = calinski_harabasz_score(newPCAdata, labels)
        db_index = davies_bouldin_score(newPCAdata,labels)
        ch.update({i: ch_index})
        db.update({i: db_index})
        
    plotMetrics(name,silhouetteVals, ch, db, distortions)

#plotMetrics function that plots silhouette values, CH index and DB index
def plotMetrics(model, silhouetteVals, ch, db, distortions=[]):
    #distortions only viable for some models
    if distortions != []:
        plt.figure(figsize=(6,4))
        plt.plot(range(2,15,1), distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The WCSS Elbow Method for ' + model)
        plt.show()
    
    plt.figure()
    plt.style.use("fivethirtyeight")
    plt.plot(range(2,15), silhouetteVals)
    plt.xticks(range(2, 15))
    plt.xlabel('k')
    plt.ylabel('Silhouette Value')
    plt.title('The Sihouette Method for ' + str(model))
    plt.show()
    
    plt.figure()
    plt.plot(list(ch.keys()), list(ch.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("Calinski-Harabasz Index")
    plt.title("CH Index for " + str(model))
    plt.show()
    
    plt.figure()
    plt.plot(list(db.keys()), list(db.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies-Bouldin Index")
    plt.title("DB Index for " + str(model))
    plt.show()
    



######################################################
######################################################
# K means 

#model KMeans for different k and show plots
model(KMeans, newPCAdata, "KMeans")
    
#fit k=3 model as best option
#record time to train
start = time.time()
km = KMeans(n_clusters=3, random_state=1)
labels = km.fit_predict(newPCAdata)
end = time.time()
print("KMeans time: " + str(end - start))


#Getting unique labels
graphKMdata['labels'] = labels
u_labels = np.unique(labels)

#plotting the results:
plt.figure(figsize=(10,6))
for j in u_labels:
    plt.scatter(graphKMdata[labels==j][0], graphKMdata[labels==j][1], label=j, s=5)
plt.legend()
plt.title("KMeans Clustering, k=3")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()


#scores for kmeans k =3
KMeansDBScore = davies_bouldin_score(newPCAdata,labels)
KMeansCHScore = calinski_harabasz_score(newPCAdata,labels)
KMeanssilScore = silhouette_score(newPCAdata, labels)


#same for k = 4 to show difference
km = KMeans(n_clusters=4, random_state=1)
labels = km.fit_predict(newPCAdata)

#Getting unique labels
graphKMdata['labels'] = labels
u_labels = np.unique(labels)

#plotting the results:
plt.figure(figsize=(10,6))
for j in u_labels:
    plt.scatter(graphKMdata[labels==j][0], graphKMdata[labels==j][1], label=j, s=5)
plt.legend()
plt.title("KMeans Clustering, k=4")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()





################################################
################################################
#mini batch k means
from sklearn.cluster import MiniBatchKMeans

model(MiniBatchKMeans, newPCAdata, "Mini-Batch KMeans")

#fit k=3 model
start = time.time()
mbk = MiniBatchKMeans(n_clusters=3, random_state=1)
labels = mbk.fit_predict(newPCAdata)
end = time.time()
print("Mini Batch Kmeans time: " + str(end - start))

#Getting unique labels
graphKMdata['labels'] = labels
u_labels = np.unique(labels)

#plotting the results:
plt.figure(figsize=(10,6))
for j in u_labels:
    plt.scatter(graphKMdata[labels==j][0], graphKMdata[labels==j][1], label=j, s=5)
plt.legend()
plt.title("Mini Batch KMeans Clustering, k=3")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()

#get scores
MiniBatchDBScore = davies_bouldin_score(newPCAdata,labels)
MiniBatchCHScore = calinski_harabasz_score(newPCAdata,labels)
MiniBatchsilScore = silhouette_score(newPCAdata, labels)




################################################
################################################
# Gaussian Mixture Model
from sklearn.mixture import GaussianMixture

#takes different parameters so cannot use model function
#do manually
#also plot BIC too

bic = []
silhouetteVals = []
ch = {}
db = {}
plt.figure()
#colours for bic lines in plot
colours = ["red", "green", "blue", "purple"]
counter = 0
#figure for bic
plt.figure(figsize=(6,4))

#different covariance matrices
for cov in ("tied","diag","spherical","full"):
    
    aic = []
    bic = []
    silhouetteVals = []
    results = {}
    db = {}
    for i in range(2,15):
        #soft kmeans method
        gm = GaussianMixture(n_components=i, random_state=1, covariance_type=cov, init_params='kmeans')
        labels = gm.fit_predict(newPCAdata)
        graphKMdata['labels'] = labels
        u_labels = np.unique(labels)
        bic.append(gm.bic(newPCAdata))
        silhouetteVals.append(silhouette_score(newPCAdata, labels))
        ch_index = calinski_harabasz_score(newPCAdata, labels)
        db_index = davies_bouldin_score(newPCAdata,labels)
        ch.update({i: ch_index})
        db.update({i: db_index})
    
    
    plt.plot(range(2,15,1), bic, 'x-',label="BIC", c=colours[counter])
    
    counter+=1

#plot BIC
plt.xlabel('Components')
plt.ylabel('BIC')
plt.title('BIC curve for different covariance matrix')    
plt.legend(["tied","diag","spherical","full"])
plt.show()

#full chosen which is last in list of covarainces, so simply plot other metrics
plotMetrics("GMM", silhouetteVals, ch, db)

#full model chosen as lowest BIC
#k=3 taken again due to BIC plot and other metrics

#fit k=3 model
start = time.time()
gm = GaussianMixture(n_components=3, random_state=1, covariance_type="full", init_params='kmeans')
labels = gm.fit_predict(newPCAdata)
end = time.time()
print("GMM time: " + str(end - start))

#Getting unique labels
graphKMdata['labels'] = labels
u_labels = np.unique(labels)

#plotting the results:
plt.figure(figsize=(10,6))
for j in u_labels:
    plt.scatter(graphKMdata[labels==j][0], graphKMdata[labels==j][1], label=j, s=5)
plt.legend()
plt.title("Gaussian Mixture Model, k=3")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()

#record scores
GMDBScore = davies_bouldin_score(newPCAdata,labels)
GMCHScore = calinski_harabasz_score(newPCAdata,labels)
GMsilScore = silhouette_score(newPCAdata, labels)





###################################################
###################################################

## Spectral Clustering
from sklearn.cluster import SpectralClustering

#uses nearest neighbours as affinity matrix to assess similarity of points
#performs spetral clustering
model(SpectralClustering, newPCAdata, "Spectral Clustering", affinity='nearest_neighbors')

#k=3 chosen again
start = time.time()
sc = SpectralClustering(n_clusters=3, random_state=1)
labels = sc.fit_predict(newPCAdata)
end = time.time()
print("Spectral time: " + str(end - start))

#Getting unique labels
graphKMdata['labels'] = labels
u_labels = np.unique(labels)

#plotting the results:
plt.figure(figsize=(10,6))
for j in u_labels:
    plt.scatter(graphKMdata[labels==j][0], graphKMdata[labels==j][1], label=j, s=5)
plt.legend()
plt.title("Spectral Clustering, k=3")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()

#get scores
SCdbScore = davies_bouldin_score(newPCAdata,labels)
SCchScore = calinski_harabasz_score(newPCAdata,labels)
SCsilScore = silhouette_score(newPCAdata, labels)

#k=4 for comparison
sc = SpectralClustering(n_clusters=4, random_state=1)
labels = sc.fit_predict(newPCAdata)

#Getting unique labels
graphKMdata['labels'] = labels
u_labels = np.unique(labels)

#plotting the results:
plt.figure(figsize=(10,6))
for j in u_labels:
    plt.scatter(graphKMdata[labels==j][0], graphKMdata[labels==j][1], label=j, s=5)
plt.legend()
plt.title("Spectral Clustering, k=4")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()







################################################
################################################
#BIRCH

from sklearn.cluster import Birch

#threshold set low- radius of the subcluster obtained by merging a new sample and the closest subcluster should be smaller than this (sklearn website)
#promotes good splitting
#could be optimised!
model(Birch,newPCAdata, "BIRCH", threshold=0.1)


#clear peak at 3 in CH index and silhouette method

start = time.time()
birch = Birch(n_clusters=3)
labels = birch.fit_predict(newPCAdata)
end = time.time()
print("BIRCH time: " + str(end - start))

#Getting unique labels
graphKMdata['labels'] = labels
u_labels = np.unique(labels)

#plotting the results:
plt.figure(figsize=(10,6))
for j in u_labels:
    plt.scatter(graphKMdata[labels==j][0], graphKMdata[labels==j][1], label=j, s=5)
plt.legend()
plt.title("BIRCH Clustering, k=3")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()

#get scores
BIRCHsilScore = silhouette_score(newPCAdata, labels)
birchDBScore = davies_bouldin_score(newPCAdata,labels)
birchCHScore = calinski_harabasz_score(newPCAdata,labels)



