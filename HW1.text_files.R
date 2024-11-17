
#The folder "text_files" contains text files written on various topics by two persons. Interests of person 1 and person 2 are very distinct. Note that the file title doesn't necessarily reflect its content.
#Use a chosen method of clustering in order to find into which categories/concepts/topics belong these files. Can you say anything relevant about types of these texts or, maybe, about their authors?

################################################################################
####### LOADING LIBRARIES
################################################################################

# Load libraries
library(textmineR)
library(stopwords)
library(wordcloud)



################################################################################
####### LOADING FILES
################################################################################

getwd()

# Load text data
text_files <- list.files("text_files", pattern = "*.txt", full.names = TRUE)
doc_names <- tools::file_path_sans_ext(basename(text_files))
doc_vec <- setNames(as.list(sapply(text_files, function(file) {
  lines <- readLines(file, warn = FALSE)
  paste(lines, collapse = " ")
})), doc_names)

str(doc_vec)

doc_vec$Address



################################################################################
####### SETTING UP DTM
################################################################################

parallel::detectCores()

# Defining additional stop words -> after checking it does not rly matter, i get the same conclusions as before but it looks better on graphs
custom_stopwords <- c("It", "He", "She", "They", "Your", "and", "they", "you", "it", "he", "she", "them", "me", "mine", "the", "And", "The", "THE") #"I", "we", "We", "Us", "us", "our", "your", 
# i left out words like We, I, Our as after checking first results i though that its related to some kind of political statements and these words are kinda important

# Create DTM 
dtm <- CreateDtm(doc_vec = doc_vec,
                 doc_names = doc_names,
                 ngram_window = c(1, 3),
                 stopword_vec = c(stopwords::stopwords("en"), stopwords::stopwords(source = "smart"),  custom_stopwords), #, custom_stopwords
                 lower = FALSE,
                 remove_punctuation = TRUE,
                 remove_numbers = FALSE,
                 stem_lemma_function = NULL,
                 cpus = 4)

# Calculate TF-IDF and cosine similarity
tf_mat <- TermDocFreq(dtm)
tfidf <- t(dtm[ , tf_mat$term ]) * tf_mat$idf
tfidf <- t(tfidf)
csim <- tfidf / sqrt(rowSums(tfidf * tfidf))
csim <- csim %*% t(csim)
cdist <- as.dist(1 - csim)



################################################################################
####### CLUSTERING AND ANALYSIS
################################################################################

# Perform hierarchical clustering
hc <- hclust(cdist, "ward.D")
num_clusters <- 2  # Expecting two authors
clustering <- cutree(hc, num_clusters)

# Plot the dendrogram
plot(hc, main = "Hierarchical Clustering of Text Files",
     ylab = "", xlab = "", yaxt = "n")
rect.hclust(hc, k = num_clusters, border = "red")

# Dendrogram does not rreally help me that much in assesning what kind of topicss each person wrote about so i will just follow further



################################################################################
####### overall word proportions and identify top words in each cluster
################################################################################

# Calculate overall word proportions and identify top words in each cluster
p_words <- colSums(dtm) / sum(dtm)
cluster_words <- lapply(unique(clustering), function(x) {
  rows <- dtm[clustering == x, ]
  rows <- rows[, colSums(rows) > 0]
  colSums(rows) / sum(rows) - p_words[colnames(rows)]
})

# Summarize each cluster with the top words
cluster_summary <- data.frame(cluster = unique(clustering),
                              size = as.numeric(table(clustering)),
                              top_words = sapply(cluster_words, function(d) {
                                paste(names(d)[order(d, decreasing = TRUE)][1:50], collapse = ", ")
                              }),
                              stringsAsFactors = FALSE)
print(cluster_summary)



################################################################################
####### Word cloud
################################################################################

# Set up a 1x2 layout for side-by-side word clouds
par(mfrow = c(1, 2))

# Generate word cloud for each cluster
for (i in 1:2) {
  wordcloud(words = names(cluster_words[[i]]), 
            freq = cluster_words[[i]], 
            max.words = 50, 
            random.order = FALSE, 
            colors = c("red", "yellow", "blue"),
            main = paste("Top Words in Cluster", i))
}

# Reset layout to default
par(mfrow = c(1, 1))

# ANSWER 
# Author 1 - cluster 1
# may be focused on socio-political subjects, given the language and topics in Cluster 1. If we know one author leans toward political themes, they likely wrote documents grouped in this cluster.
# (America, I, We, United States, Clinton, Hillary, Obamacare, etc.)
# Author 2 - Cluster 2 
# likely specializes in business and technical topics, aligning with the terminology in Cluster 2, which suggests a more structured, professional focus on data and organizational theory.
# (managment, data, aproach, decision, processes, etc.)