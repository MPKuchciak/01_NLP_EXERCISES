
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

# Create histograms for distribution
library(ggplot2)



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

# Setting seed
set.seed(420)



################################################################################
###### Functions for DTM
################################################################################

# Apply lemmatization - function
lemma_func <- function(words) {
  textstem::lemmatize_words(words)  # lemmatize each word in the input vector
}

# Apply stemming - function
stemmed_func <- function(words) {
  SnowballC::wordStem(words, language = "en")  # steam each word in the input vector
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
                                     ngram_window = c(1, 2), # n-gram window for unigrams and bigrams
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
                                ngram_window = c(1, 2), # n-gram window for unigrams and bigrams
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
# Also checked ngram 1:2 and 1:3 and it does not change much in out dtm df's



################################################################################
####### COSINE SIMILARITY AND DISTANCE MATRIX
################################################################################

dtm <- dtm_lemmatization_stand

# Compute TF-IDF and cosine similarity
tf_mat <- TermDocFreq(dtm)
tfidf <- t(dtm[, tf_mat$term]) * tf_mat$idf
tfidf <- t(tfidf)

csim <- tfidf / sqrt(rowSums(tfidf * tfidf))

csim <- tfidf / sqrt(rowSums(tfidf * tfidf) + 1e-8) # Add a small constant to avoid division by zero
csim <- csim %*% t(csim)
cdist <- as.dist(1 - csim)



################################################################################
####### checking dtm, termdocfreq
################################################################################

# Summary statistics for `doc_freq` and `idf`
summary_stats <- data.frame(
  Metric = c("Doc Frequency", "IDF"),
  Min = c(min(tf_mat$doc_freq), min(tf_mat$idf)),
  Mean = c(mean(tf_mat$doc_freq), mean(tf_mat$idf)),
  Median = c(median(tf_mat$doc_freq), median(tf_mat$idf)),
  Max = c(max(tf_mat$doc_freq), max(tf_mat$idf)),
  SD = c(sd(tf_mat$doc_freq), sd(tf_mat$idf))
)
print(summary_stats)

# Document Frequency
ggplot(tf_mat, aes(x = doc_freq)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Document Frequency", x = "Document Frequency", y = "Count") +
  theme_minimal()

# IDF
ggplot(tf_mat, aes(x = idf)) +
  geom_histogram(binwidth = 0.5, fill = "coral", color = "black") +
  labs(title = "Distribution of Inverse Document Frequency (IDF)", x = "IDF", y = "Count") +
  theme_minimal()

# Sort terms by TF, DF, and IDF
term_stats <- tf_mat
str(term_stats)
# Sort terms by Term Frequency (TF)
term_stats_sorted_tf <- term_stats[order(-term_stats$term_freq), ]

# Sort terms by Document Frequency (DF)
term_stats_sorted_df <- term_stats[order(-term_stats$doc_freq), ]

# Sort terms by Inverse Document Frequency (IDF)
term_stats_sorted_idf <- term_stats[order(-term_stats$idf), ]

# Check the sorted data frames
head(term_stats_sorted_tf)
head(term_stats_sorted_df)
head(term_stats_sorted_idf)



################################################################################
####### COSINE SIMILARITY AND DISTANCE MATRIX - FILTERED VERSION
################################################################################

# Filtering terms with doc_freq < 5 or doc_freq > 0.8 * number of documents
min_doc_freq <- 2

#max_doc_freq <- 0.8 * nrow(doc_vec)
max_doc_freq <- length(doc_vec)

tf_mat_filtered <- tf_mat[tf_mat$doc_freq >= min_doc_freq & tf_mat$doc_freq <= max_doc_freq, ]

nrow(tf_mat) - nrow(tf_mat_filtered)
nrow(tf_mat_filtered)

# Apply the filter to DTM
dtm_filtered <- dtm[, tf_mat_filtered$term]



################################################################################

# Compute cosine similarity and distance matrix
tfidf_filtered <- t(dtm_filtered[, tf_mat_filtered$term]) * tf_mat_filtered$idf
tfidf_filtered <- t(tfidf_filtered)

# Normalize and calculate cosine similarity
csim_filtered <- tfidf_filtered / sqrt(rowSums(tfidf_filtered * tfidf_filtered) + 1e-8)
csim_filtered <- csim_filtered %*% t(csim_filtered)
cdist_filtered <- as.dist(1 - csim_filtered)



################################################################################
####### CLUSTERING #############################################################
################################################################################
################################################################################
####### STEP 1.1: HIERARCHICAL CLUSTERING
###############################################################################

# Perform hierarchical clustering -> may take a long time for n-grams = 3
hc <- hclust(cdist, method = "ward.D")
# Perform hierarchical clustering for cdist filtered
hc_filtered <- hclust(cdist_filtered, method = "ward.D")

#plot(hc_filtered, labels = FALSE, main = "Hierarchical Clustering Dendrogram")



################################################################################
####### STEP 1.2: SILHOUETTE ANALYSIS
################################################################################

# ------------ HC --------------------------------------------------------------
# Silhouette scores for k = 2 to 10
silhouette_scores_hc <- sapply(2:10, function(k) {
  clusters <- cutree(hc, k)
  mean(silhouette(clusters, cdist)[, 3])  # Average silhouette width
})

# Plot silhouette scores
plot(2:10, silhouette_scores_hc, type = "b", xlab = "Number of Clusters (k)",
     ylab = "Average Silhouette Width", 
     main = "Silhouette Analysis for Hierarchical Clustering",
     col = "blue", pch = 16)

# Optimal number of clusters based on silhouette scores ( we do not consider highest number of clusters where values would be highest beacause so many topics do not make sense for answering question given in a task)
optimal_k_silhouette <- which.max(silhouette_scores_hc) + 1
cat("Optimal number of clusters (Silhouette):", optimal_k_silhouette, "\n")



# ------------ HC_filtered -----------------------------------------------------
silhouette_scores_hc_filtered <- sapply(2:10, function(k) {
  clusters <- cutree(hc_filtered, k)
  mean(silhouette(clusters, cdist_filtered)[, 3])  # Average silhouette width
})

# Plot silhouette scores
plot(2:10, silhouette_scores_hc_filtered, type = "b", xlab = "Number of Clusters (k)",
     ylab = "Average Silhouette Width", 
     main = "Silhouette Analysis for Hierarchical Clustering",
     col = "blue", pch = 16)

# Optimal number of clusters based on silhouette scores ( we do not consider highest number of clusters)
optimal_k_silhouette_filtered <- which.max(silhouette_scores_hc_filtered) + 1
cat("Optimal number of clusters (Silhouette):", optimal_k_silhouette_filtered, "\n")



################################################################################
####### STEP 1.3: ELBOW METHOD
################################################################################

# ------------ wcss_hc ---------------------------------------------------------
# Convert cdist to a matrix for indexing
cdist_matrix <- as.matrix(cdist)

# Compute WCSS for k = 2 to 10
wcss_hc <- sapply(2:10, function(k) {
  clusters <- cutree(hc, k)
  sum(sapply(unique(clusters), function(cluster) {
    cluster_indices <- which(clusters == cluster)
    cluster_dists <- cdist_matrix[cluster_indices, cluster_indices]
    sum(cluster_dists^2) / length(cluster_indices)
  }))
})

# Plot WCSS for hierarchical clustering
plot(2:10, wcss_hc, type = "b", xlab = "Number of Clusters (k)",
     ylab = "WCSS", main = "Elbow Method for Hierarchical Clustering",
     col = "red", pch = 16)

# Convert WCSS data into a data frame
elbow_data <- data.frame(
  k = 2:10,
  WCSS = wcss_hc
)

# Identify the elbow point (approximation)
elbow_point <- which.min(diff(wcss_hc)) + 1  # Adjust as needed
ggplot(elbow_data, aes(x = k, y = WCSS)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 3) +
  geom_point(data = elbow_data[elbow_data$k == elbow_point, ],
             aes(x = k, y = WCSS), color = "green", size = 5) +
  labs(
    title = "Elbow Method for Hierarchical Clustering",
    x = "Number of Clusters (k)",
    y = "WCSS (Within-Cluster Sum of Squares)"
  ) +
  theme_minimal() +
  annotate("text", x = elbow_point, y = wcss_hc[elbow_point],
           label = paste("Optimal k =", elbow_point), vjust = -1.5, color = "darkgreen")



# ------------ wcss_hc_filtered ------------------------------------------------
# Convert cdist to a matrix for indexing
cdist_matrix_filtered <- as.matrix(cdist_filtered)

# Compute WCSS for k = 2 to 10
wcss_hc_filtered <- sapply(2:10, function(k) {
  clusters <- cutree(hc_filtered, k)
  sum(sapply(unique(clusters), function(cluster) {
    cluster_indices <- which(clusters == cluster)
    cluster_dists <- cdist_matrix_filtered[cluster_indices, cluster_indices]
    sum(cluster_dists^2) / length(cluster_indices)
  }))
})

# Plot WCSS for hierarchical clustering
plot(2:10, wcss_hc_filtered, type = "b", xlab = "Number of Clusters (k)",
     ylab = "WCSS", main = "Elbow Method for Hierarchical Clustering",
     col = "red", pch = 16)

# Convert WCSS data into a data frame
elbow_data <- data.frame(
  k = 2:10,
  WCSS = wcss_hc_filtered
)

# Identify the elbow point (approximation)
elbow_point <- which.min(diff(wcss_hc_filtered)) + 1  # Adjust as needed
ggplot(elbow_data, aes(x = k, y = WCSS)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 3) +
  geom_point(data = elbow_data[elbow_data$k == elbow_point, ],
             aes(x = k, y = WCSS), color = "green", size = 5) +
  labs(
    title = "Elbow Method for Hierarchical Clustering",
    x = "Number of Clusters (k)",
    y = "WCSS (Within-Cluster Sum of Squares)"
  ) +
  theme_minimal() +
  annotate("text", x = elbow_point, y = wcss_hc_filtered[elbow_point],
           label = paste("Optimal k =", elbow_point), vjust = -1.5, color = "darkgreen")


# K = 2 for most methods , 3 for none



################################################################################
####### STEP 1.4: EVALUATE CLUSTERS FOR K = 2:10
################################################################################

# ------------ non-filtered ----------------------------------------------------
# Evaluate clusters and store cluster memberships for k = 2 to 10
cluster_results <- list()

for (k in 2:10) {
  cluster_results[[k]] <- cutree(hc, k)
  cat(sprintf("Cluster assignments for k = %d:\n", k))
  print(table(cluster_results[[k]]))  # Frequency of elements in each cluster
}



# ------------ filtered --------------------------------------------------------
# Evaluate clusters and store cluster memberships for k = 2 to 10
cluster_results <- list()

for (k in 2:10) {
  cluster_results[[k]] <- cutree(hc_filtered, k)
  cat(sprintf("Cluster assignments for k = %d:\n", k))
  print(table(cluster_results[[k]]))  # Frequency of elements in each cluster
}

# filtered seems better for 2 and unfiltered for 3



################################################################################
####### STEP 1.5: USE OPTIMAL CLUSTERING RESULT
################################################################################

# ------------ non-filtered ----------------------------------------------------
# Use the optimal number of clusters based on silhouette analysis
optimal_clusters <- cutree(hc, optimal_k_silhouette)

# Output cluster memberships for the optimal k
cat("\nOptimal clusters based on silhouette scores:\n")
print(table(optimal_clusters))  # Frequency of elements in each cluster



# ------------ filtered --------------------------------------------------------
# Use the optimal number of clusters based on silhouette analysis
optimal_clusters <- cutree(hc_filtered, optimal_k_silhouette_filtered)

# Output cluster memberships for the optimal k
cat("\nOptimal clusters based on silhouette scores:\n")
print(table(optimal_clusters))  # Frequency of elements in each cluster



################################################################################
####### STEP 2.1: K-MEANS CLUSTERING
################################################################################\

# Define a range for the number of clusters - takes a lot of time so adjust according to previous method
k_range <- 2:5



################################################################################
# 2.2: Function for Silhouette and Elbow Analysis
################################################################################
kmeans_analysis <- function(tfidf_matrix, k_range) {
  # Initialize vectors for results
  silhouette_scores <- numeric(length(k_range))
  wcss <- numeric(length(k_range))
  
  for (i in seq_along(k_range)) {
    k <- k_range[i]
    kmeans_result <- kmeans(tfidf_matrix, centers = k, nstart = 25)
    wcss[i] <- kmeans_result$tot.withinss
    
    # Compute silhouette score only if k > 1
    if (k > 1) {
      silhouette_scores[i] <- mean(silhouette(kmeans_result$cluster, dist(tfidf_matrix))[, 3])
    } else {
      silhouette_scores[i] <- NA
    }
  }
  
  # Silhouette Plot
  plot(k_range, silhouette_scores, type = "b", pch = 16, col = "blue",
       xlab = "Number of Clusters (k)", ylab = "Average Silhouette Score",
       main = "Silhouette Analysis for Optimal k")
  abline(v = k_range[which.max(silhouette_scores)], col = "red", lty = 2)
  
  # Elbow Plot
  plot(k_range, wcss, type = "b", pch = 16, col = "red",
       xlab = "Number of Clusters (k)", ylab = "WCSS (Within-Cluster Sum of Squares)",
       main = "Elbow Method for Optimal k")
  abline(v = which.min(abs(diff(wcss))), col = "blue", lty = 2)
  
  # Return optimal k
  list(
    optimal_k_silhouette = k_range[which.max(silhouette_scores)],
    elbow_point = k_range[which.min(abs(diff(wcss)))]
    )
}



################################################################################
# find optimal k on k-means, both on filtered and unfiltered tfidf
################################################################################

# Define range of k
k_range <- 2:5

# Run the function
results <- kmeans_analysis(tfidf, k_range)

# Print optimal k
cat("Optimal k (Silhouette):", results$optimal_k_silhouette, "\n")
cat("Optimal k (Elbow Point):", results$elbow_point, "\n")

# on graphical analaysis its more like K is 3

# Define range of k
k_range <- 2:5

# Run the function
results_filtered <- kmeans_analysis(tfidf_filtered, k_range)

# Print optimal k
cat("Optimal k (Silhouette):", results_filtered$optimal_k_silhouette, "\n")
cat("Optimal k (Elbow Point):", results_filtered$elbow_point, "\n") # should be 3



################################################################################
####### STEP 3: ANALYZE AND NAME CLUSTERS
################################################################################

# Perform hierarchical clustering (Ward's method)
hc <- hclust(cdist, "ward.D")

# Cut the dendrogram to get a desired number of clusters
clustering <- cutree(hc, 2)  # Adjust the number of clusters 

k <- 2  # Number of clusters
km <- kmeans(cdist, centers = k, nstart = 100)  # Run K-means clustering



################################################################################
###### FURTHER PROCESSING
################################################################################

k <- 2  # Number of clusters for K-means

# Load necessary libraries
library(slam)
library(cluster)
library(wordcloud)

# Term frequencies for DTM and TF-IDF
term_freq_dtm <- slam::col_sums(dtm)
term_freq_tfidf <- slam::col_sums(tfidf)

# Check term frequencies
print(term_freq_dtm)
print(term_freq_tfidf)

# Cluster summaries for K-means clustering
cluster_summary <- lapply(1:k, function(cluster_num) {
  cluster_indices <- which(km$cluster == cluster_num)
  cluster_dtm <- dtm[cluster_indices, , drop = FALSE]
  
  if (length(dim(cluster_dtm)) > 1) {
    term_freq <- slam::col_sums(cluster_dtm)
  } else {
    term_freq <- cluster_dtm
  }
  
  top_terms <- names(sort(term_freq, decreasing = TRUE)[1:20])
  
  list(cluster = cluster_num, top_terms = top_terms)
})

# Print cluster summaries
print(cluster_summary)



################################################################################
###### HIERARCHICAL CLUSTERING (NON-FILTERED)
################################################################################

hc <- hclust(cdist, "ward.D")
clustering <- cutree(hc, k)

# Inspect clusters using the probability difference method
p_words <- col_sums(dtm) / sum(dtm)

cluster_words <- lapply(unique(clustering), function(cluster_id) {
  rows <- dtm[clustering == cluster_id, ]
  rows <- rows[, col_sums(rows) > 0]
  
  col_sums(rows) / sum(rows) - p_words[colnames(rows)]
})

# Create summary of top words defining each cluster
hc_summary <- data.frame(
  cluster = unique(clustering),
  size = as.numeric(table(clustering)),
  top_words = sapply(cluster_words, function(d) {
    paste(names(d)[order(d, decreasing = TRUE)][1:5], collapse = ", ")
  }),
  stringsAsFactors = FALSE
)

# Print hierarchical clustering summary
print(hc_summary)

# Plot a word cloud for one cluster (e.g., first cluster as an example)
example_cluster <- 1
if (example_cluster <= length(cluster_words)) {
  wordcloud::wordcloud(
    words = names(cluster_words[[example_cluster]]),
    freq = cluster_words[[example_cluster]],
    max.words = 50,
    random.order = FALSE,
    colors = c("red", "yellow", "blue"),
    main = paste("Top words in cluster", example_cluster)
  )
}

################################################################################
###### HIERARCHICAL CLUSTERING (FILTERED)
################################################################################

hc_filtered <- hclust(cdist_filtered, "ward.D")
clustering_filtered <- cutree(hc_filtered, k)

# Inspect filtered clusters using the probability difference method
cluster_words_filtered <- lapply(unique(clustering_filtered), function(cluster_id) {
  rows <- dtm[clustering_filtered == cluster_id, ]
  rows <- rows[, col_sums(rows) > 0]
  
  col_sums(rows) / sum(rows) - p_words[colnames(rows)]
})

# Create summary of top words for filtered clustering
hc_filtered_summary <- data.frame(
  cluster = unique(clustering_filtered),
  size = as.numeric(table(clustering_filtered)),
  top_words = sapply(cluster_words_filtered, function(d) {
    paste(names(d)[order(d, decreasing = TRUE)][1:5], collapse = ", ")
  }),
  stringsAsFactors = FALSE
)

# Print filtered hierarchical clustering summary
print(hc_filtered_summary)

################################################################################
###### K-MEANS CLUSTERING
################################################################################

kfit <- kmeans(cdist, centers = 5, nstart = 100)

# Plot K-means clustering results
clusplot(as.matrix(cdist), kfit$cluster, color = TRUE, shade = TRUE, labels = 2, lines = 0)
