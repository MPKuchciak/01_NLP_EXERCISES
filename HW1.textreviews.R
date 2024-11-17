
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

# 1. DTM Creation
dtm <- CreateDtm(doc_vec = doc_vec,
                 doc_names = doc_names,
                 ngram_window = c(1, 2),
                 stopword_vec = stopword_vec, 
                 lower = TRUE,
                 remove_punctuation = TRUE,
                 remove_numbers = TRUE,
                 cpus = 4)

# 2. Apply TF-IDF weighting
tf_mat <- TermDocFreq(dtm)
tfidf <- t(dtm[, tf_mat$term]) * tf_mat$idf
tfidf <- t(tfidf)

head(tfidf)

str(dtm)
str(tfidf) #bugs? -> sparse matrix?

# Check the cleaned text for the first few reviews
head(doc_vec, 10)

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