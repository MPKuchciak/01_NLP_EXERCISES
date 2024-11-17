
# Exercise text
# The dataset textreview.csv consists of 2000 text reviews concerning products of more than one distinct industry branches.
# Use a chosen method of clustering to find and name these main industry branches.

################################################################################
####### LOADING LIBRARIES
################################################################################

# Load libraries - some libraries may be loaded later on 
library(textmineR)
library(stopwords)
library(wordcloud)



################################################################################
####### LOADING FILES
################################################################################

data_folder <- "TASK_02_DATA"

# Set working directory and load data eventuially use Rproj
getwd()
#setwd()

# Load and preprocess data
reviews <- read.csv(file.path("00_DATA", data_folder, "textreviews.csv"), stringsAsFactors = FALSE)

# Extract the text column and create document names
doc_vec <- reviews$text
doc_names <- paste0("review_", reviews$id)

# Verify the setup
str(doc_vec)
str(doc_names)



################################################################################
###### FURTHER PROCESSING
################################################################################

length(doc_vec) 
length(doc_names) 

sum(is.na(doc_vec))         # Should be 0
sum(nchar(doc_vec) == 0)    # Should be 0
# BUT, # Error in nchar(doc_vec) : invalid multibyte string, element 1298

# we will go through few steps
# Step 1: Convert with a placeholder for non-UTF-8 characters
doc_vec <- iconv(doc_vec, from = "latin1", to = "UTF-8", sub = "byte")

sum(nchar(doc_vec) == 0)    # Should be 0
doc_vec[nchar(doc_vec) == 0]
# now we are left with empty strings wchich do not help us later on within clustering or using models so we will remove them

# Step 2: Verify empty or problematic entries left (location of these rows - indexes)
empty_docs <- which(nchar(doc_vec) == 0)
print(empty_docs)

# Step 3: Identify non-empty entries and their removal from both doc_names and doc_vec
non_empty_docs <- nchar(doc_vec) > 0
doc_vec <- doc_vec[non_empty_docs]
doc_names <- doc_names[non_empty_docs]

# final check
sum(is.na(doc_vec))         # Should be 0
sum(nchar(doc_vec) == 0)    # Should be 0
# and we are done, we will proceed to set up parameters of dtm 

# Update stopwords and clean text
stopword_vec <- c(stopwords::stopwords("en"), stopwords::stopwords(source = "smart"))
stopword_vec <- unique(stopword_vec)


################################################################################
###### DTM
################################################################################

# Load necessary libraries
library(textstem)  # For lemmatization
library(SnowballC) # For stemming

# Apply lemmatization
lemma_func <- function(words) {
  textstem::lemmatize_words(words)  # lemmatize each word in the input vector
}
  
# Apply stemming
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
dtm_standard <- CreateDtm(
  doc_vec = doc_vec, 
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
head(sort(colSums(as.matrix(dtm_lemmatization_stand)), decreasing = TRUE), 10) # this one is choosen
head(sort(colSums(as.matrix(dtm_stemming_stand)), decreasing = TRUE), 10)

# I will go with 3rd aproach as It does not take so long and its more accurate and precise because it considers context and part of speech.



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
# As we removed problematic rows earlier we do not have to perform any additional modifications like adding small number

#csim <- tfidf / sqrt(rowSums(tfidf * tfidf) + 1e-8)  # Add a small constant to avoid division by zero
#csim <- csim %*% t(csim)  # Calculate cosine similarity matrix
#cdist <- as.dist(1 - csim)  # Convert cosine similarity to distance



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