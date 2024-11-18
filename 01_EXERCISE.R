
# Exercise text
# The dataset textreview.csv consists of 2000 text reviews concerning products of more than one distinct industry branches.
# Use a chosen method of clustering to find and name these main industry branches.

################################################################################
####### LOADING LIBRARIES
################################################################################

# Load libraries - some libraries may be loaded later on 
library(textmineR)
library(stopwords)
library(cluster)

library(textstem)  # For lemmatization
library(SnowballC) # For stemming



################################################################################
####### LOADING FILES
################################################################################

# Load data
data_folder <- "TASK_02_DATA"
reviews <- read.csv(file.path("00_DATA", data_folder, "textreviews.csv"), stringsAsFactors = FALSE)

# Extract text and create document names
doc_vec <- iconv(reviews$text, from = "latin1", to = "UTF-8", sub = "byte")
doc_names <- paste0("review_", reviews$id)

# Remove empty "documents" ; reviews
non_empty_docs <- nchar(doc_vec) > 0
doc_vec <- doc_vec[non_empty_docs]
doc_names <- doc_names[non_empty_docs]

# Stopword setup
stopword_vec <- unique(c(stopwords::stopwords("en"), stopwords::stopwords(source = "smart")))

# Verify the setup; both vectors should be of the same number of elements
str(doc_vec)
str(doc_names)



################################################################################
###### Functions for DTM
################################################################################

# Apply lemmatization - function
lemma_func <- function(words) {
  textstem::lemmatize_words(words)  # lemmatize each word in the input vector
}

# Apply stemming - function
stemmed_func <- function(words) {
  SnowballC::wordStem(words, language = "en")  # lemmatize each word in the input vector
}

################################################################################
# 1. DTM Creation - Create the document-term matrix (DTM)

# 1.1. No Preprocessing (Baseline)
# A DTM with raw text:
#   - No punctuation removal
#   - No number removal
#   - No stemming or lemmatization
#   - case-sensitive
dtm_baseline <- CreateDtm(doc_vec = doc_vec, # character vector of documents
                          doc_names = doc_names, # document names
                          ngram_window = c(1, 2), # n-gram window for unigrams and bigrams
                          stopword_vec = stopword_vec, # English stopwords
                          lower = FALSE, # Convert to lowercase
                          remove_punctuation = FALSE, # Remove punctuation
                          remove_numbers = FALSE, # Remove numbers
                          verbose = FALSE, # Turn off progress bar
                          cpus = 4, # Use X CPUs 
                          stem_lemma_function = NULL) #lemma_func

# 1.2. Standard Preprocessing
# A DTM with common preprocessing:
#   - Convert text to lowercase
#   - Remove punctuation and numbers
#   - Apply stopwords
dtm_standard <- CreateDtm(doc_vec = doc_vec, 
  doc_names = doc_names, 
  ngram_window = c(1, 2), 
  stopword_vec = stopword_vec, 
  lower = TRUE, 
  remove_punctuation = TRUE, 
  remove_numbers = TRUE, 
  verbose = FALSE, 
  cpus = 4, 
  stem_lemma_function = NULL
)

# 1.3. Standard Preprocessing + lemmatization
# A DTM with common preprocessing:
#   - Convert text to lowercase
#   - Remove punctuation and numbers
#   - Apply stopwords
#   - Apply lemmatization
dtm_lemmatization_stand <- CreateDtm(doc_vec = doc_vec, # character vector of documents
                                     doc_names = doc_names, # document names
                                     ngram_window = c(1, 3), # n-gram window for unigrams and bigrams
                                     stopword_vec = stopword_vec, # English stopwords
                                     lower = TRUE, # Convert to lowercase
                                     remove_punctuation = TRUE, # Remove punctuation
                                     remove_numbers = TRUE, # Remove numbers
                                     verbose = FALSE, # Turn off progress bar
                                     cpus = 4, # Use X CPUs 
                                     stem_lemma_function = lemma_func) #lemma_func

# 1.4. Standard Preprocessing
# A DTM with common preprocessing:
#   - Convert text to lowercase
#   - Remove punctuation and numbers
#   - Apply stopwords
#   - Apply stemming
dtm_stemming_stand <- CreateDtm(doc_vec = doc_vec, # character vector of documents
                                doc_names = doc_names, # document names
                                ngram_window = c(1, 3), # n-gram window for unigrams and bigrams
                                stopword_vec = stopword_vec, # English stopwords
                                lower = TRUE, # Convert to lowercase
                                remove_punctuation = TRUE, # Remove punctuation
                                remove_numbers = TRUE, # Remove numbers
                                verbose = FALSE, # Turn off progress bar
                                cpus = 4, # Use X CPUs 
                                stem_lemma_function = stemmed_func) #lemma_func


head(sort(colSums(as.matrix(dtm_standard)), decreasing = TRUE), 10)
head(sort(colSums(as.matrix(dtm_baseline)), decreasing = TRUE, ), 10)
head(sort(colSums(as.matrix(dtm_lemmatization_stand)), decreasing = TRUE), 10) # this one is the one selected
head(sort(colSums(as.matrix(dtm_stemming_stand)), decreasing = TRUE), 10)

# I will go with 3rd aproach as It does not take so long and its more accurate and precise because it considers context and part of speech.



################################################################################
####### COSINE SIMILARITY AND DISTANCE MATRIX
################################################################################

# Compute TF-IDF and cosine similarity
tf_mat <- TermDocFreq(dtm)
tfidf <- t(dtm[, tf_mat$term]) * tf_mat$idf
tfidf <- t(tfidf)

csim <- tfidf / sqrt(rowSums(tfidf * tfidf))

csim <- tfidf / sqrt(rowSums(tfidf * tfidf) + 1e-8) # Add a small constant to avoid division by zero
csim <- csim %*% t(csim)
cdist <- as.dist(1 - csim)



################################################################################
####### ENTRY TO CLUSTERING
################################################################################

# Perform hierarchical clustering
hc <- hclust(cdist, "ward.D")

# Determine optimal number of clusters (k) visually using a dendrogram -> impossible from this kind of data
plot(hc)  # Review the plot and decide the number of clusters


# 1. Elbow Method Using Sum of Squared Distances
# Compute WSS for k = 1 to 10
wss <- sapply(1:10, function(k) {
  cluster <- cutree(hc, k)  # Cut dendrogram into k clusters
  sum(sapply(unique(cluster), function(c) {
    cluster_points <- tfidf[cluster == c, ]
    cluster_center <- colMeans(cluster_points)
    sum(rowSums((cluster_points - cluster_center)^2))
  }))
})

# Plot WSS
plot(1:10, wss, type = "b", xlab = "Number of Clusters (k)", ylab = "Within-Cluster Sum of Squares")

#2. Silhouette Analysis
silhouette_scores <- sapply(2:10, function(k) {
  cluster <- cutree(hc, k)
  mean(silhouette(cluster, cdist)[, 3])  # Average silhouette width
})

# Plot silhouette scores
plot(2:10, silhouette_scores, type = "b", xlab = "Number of Clusters (k)", ylab = "Average Silhouette Width")

# 3.
library(cluster)
set.seed(42)

gap_stat <- clusGap(as.matrix(tfidf), FUN = hclust, K.max = 10, B = 50)
plot(gap_stat, main = "Gap Statistic")
optimal_k <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"])
print(optimal_k)  # Optimal number of clusters

#4. 
set.seed(42)
k <- 3  # Replace with the desired number of clusters or use methods above to determine
kmeans_result <- kmeans(tfidf, centers = k, nstart = 25)

# Assign cluster labels
clusters <- kmeans_result$cluster

# Check cluster sizes
table(clusters)




#5.
library(ggplot2)

# Reduce dimensions using PCA
pca_result <- prcomp(tfidf, center = TRUE, scale. = TRUE)
pca_data <- data.frame(pca_result$x[, 1:2])  # Use first 2 principal components

# Perform clustering on reduced data
k <- 3  # Number of clusters
kmeans_result <- kmeans(pca_data, centers = k, nstart = 25)

# Visualize clusters
pca_data$cluster <- factor(kmeans_result$cluster)
ggplot(pca_data, aes(x = PC1, y = PC2, color = cluster)) + 
  geom_point() + 
  theme_minimal() + 
  labs(title = "Clusters Visualized on PCA Reduced Data")




#6.
set.seed(42)

# Determine the optimal k using the elbow method
wss <- sapply(1:10, function(k) {
  kmeans(tfidf, centers = k, nstart = 25)$tot.withinss
})

plot(1:10, wss, type = "b", xlab = "Number of Clusters", ylab = "Total Within-Cluster Sum of Squares")

# Perform k-means with chosen k
k <- 3  # Replace with optimal k
kmeans_result <- kmeans(tfidf, centers = k, nstart = 25)
clusters <- kmeans_result$cluster


#7.

library(dbscan)

# Perform DBSCAN (eps and minPts should be tuned based on the dataset)
dbscan_result <- dbscan(as.matrix(tfidf), eps = 0.5, minPts = 10)

# Assign clusters
clusters <- dbscan_result$cluster
table(clusters)  # Check cluster sizes


#8.

library(mclust)

# Fit Gaussian mixture model
mclust_result <- Mclust(as.matrix(tfidf))

# Optimal number of clusters
optimal_k <- mclust_result$G

# Cluster assignments
clusters <- mclust_result$classification


#9.
library(cluster)

agnes_result <- agnes(as.matrix(tfidf), method = "ward")
plot(agnes_result)

# Cut into clusters
clusters <- cutree(as.hclust(agnes_result), k = 3)


k <- 3    # Replace with the desired number of clusters

# Cut dendrogram into k clusters
clusters <- cutree(hc, k)
################################################################################
###### TF-IDF + other calculations
################################################################################

# 2. Create a TF-IDF matrix
dtm <- dtm_lemmatization_stand

tf_mat <- TermDocFreq(dtm)

# Compute TF-IDF scores
tfidf <- t(dtm[, tf_mat$term]) * tf_mat$idf
tfidf <- t(tfidf)

str(dtm)
str(tfidf)

# Check the cleaned text for the first few reviews
head(doc_vec, 10)

# Cosine similarity and distance
csim <- tfidf / sqrt(rowSums(tfidf * tfidf))  # Normalizing the TF-IDF matrix
csim <- csim %*% t(csim)  # Calculate cosine similarity matrix
cdist <- as.dist(1 - csim)  # Convert cosine similarity to distance

anyNA(cdist)        # Checks for NA values
any(is.nan(cdist))  # Checks for NaN values
any(is.infinite(cdist))  # Checks for Inf values

# Identify the indices of NA and NaN values in cdist
na_indices <- which(is.na(cdist), arr.ind = TRUE)
nan_indices <- which(is.nan(cdist), arr.ind = TRUE)

# Display the rows and columns containing NA values
na_indices
nan_indices

csim <- tfidf / sqrt(rowSums(tfidf * tfidf) + 1e-8)  # Add a small constant to avoid division by zero
csim <- csim %*% t(csim)  # Calculate cosine similarity matrix
cdist <- as.dist(1 - csim)  # Convert cosine similarity to distance



# Perform hierarchical clustering (Ward's method)
hc <- hclust(cdist, "ward.D")

# Cut the dendrogram to get a desired number of clusters
clustering <- cutree(hc, 3)  # Adjust the number of clusters (5 here for example)

################################################################################
###### FURTHER PROCESSING
################################################################################

# Set seed 
set.seed(123)

k <- 7

# Load the slam package for sparse matrix 
library(slam)

# For DTM term frequency (using slam::col_sums for sparse matrices)
term_freq_dtm <- col_sums(dtm)

# For TF-IDF term frequency (using slam::col_sums for sparse matrices)
term_freq_tfidf <- col_sums(tfidf)

# cHECCK
print(term_freq_dtm)
print(term_freq_tfidf)

cluster_indices <- which(km$cluster == 1)

# Print first few cluster indices
head(cluster_indices)

cluster_summary <- lapply(1:k, function(cluster_num) {
  # Extract rows corresponding to this cluster
  cluster_indices <- which(km$cluster == cluster_num)
  cluster_dtm <- dtm[cluster_indices, , drop = FALSE]  # Ensures the result remains a matrix
  
  # Check if the matrix has more than one dimension
  if (length(dim(cluster_dtm)) > 1) {
    # Calculate term frequencies for the cluster using slam::col_sums
    term_freq <- slam::col_sums(cluster_dtm)
  } else {
    # If only one document in the cluster, calculate term frequency manually
    term_freq <- cluster_dtm
  }
  
  # Extract the top 10 terms for this cluster
  top_terms <- names(sort(term_freq, decreasing = TRUE)[1:20])
  
  list(cluster = cluster_num, top_terms = top_terms)
})

# Print cluster summaries
print(cluster_summary)

# ANSWER 
# Cluster 1: May be Finance (based on terms like "money", "fake", "credits").
# Cluster 2: May be Mobile Gaming (based on terms like "game", "play", "love").
# Cluster 3: May be Gaming (based on terms like "game", "good", "amazing").
# Cluster 4: May be Fashion (based on terms like "size", "dress", "fit", "color", "fabric").
# Cluster 5: May be Gaming or Lifestyle (based on terms like "game", "love", "fun", "addictive").
# Cluster 6: May be Lifestyle (based on terms like "amazing", "comfortable", "absolute").
# Cluster 7: May be Lifestyle or Fashion (terms like "amazing", "comfortable", "absolutely").
# Cluster 8: May be Fashion (terms like "perfect", "size", "dress").
# Cluster 9: May be Fashion or Goods (terms like "adorable", "perfect", "comfortable").
# Cluster 10: May be Lifestyle or E-commerce (terms like "perfect", "comfortable", "amazing").


# preferably should use less clusters than 10 beacause some clusters overlapp, I had some issues with this sparse matrix 