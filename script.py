import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    try:
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
        return ' '.join(words)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

## we load and preprocess all text files in the specified directory, 
def load_and_preprocess_texts(directory):
    preprocessed_texts = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                preprocessed_text = preprocess_text(text)
                if preprocessed_text:
                    preprocessed_texts.append(preprocessed_text)
                    filenames.append(filename)
    return preprocessed_texts, filenames

# We try to vectorize the texts using TF-IDF
def tfidf_vectorize(texts):
    try:
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix, vectorizer
    except Exception as e:
        print(f"Error in TF-IDF vectorization: {e}")
        return None, None


def word2vec_vectorize(texts):
    try:
        tokenized_texts = [word_tokenize(text) for text in texts]
        model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
        word2vec_matrix = np.array([np.mean([model.wv[word] for word in words if word in model.wv]
                                            or [np.zeros(100)], axis=0) for words in tokenized_texts])
        return word2vec_matrix, model
    except Exception as e:
        print(f"Error in Word2Vec vectorization: {e}")
        return None, None



## we evaluate the clustering performance using silhouette score and Davies-Bouldin score
def evaluate_clustering(matrix, method='kmeans', max_clusters=10):
    best_n_clusters = 2
    best_score = -1
    best_labels = None
    
    for n_clusters in range(2, max_clusters + 1):
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=0).fit(matrix)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters).fit(matrix)
        else:
            raise ValueError("Unsupported clustering method.")
        
        labels = model.labels_
        silhouette_avg = silhouette_score(matrix, labels)
        if silhouette_avg > best_score:
            best_n_clusters = n_clusters
            best_score = silhouette_avg
            best_labels = labels

    db_score = davies_bouldin_score(matrix, best_labels)
    
    return best_n_clusters, best_score, db_score, best_labels


# Visulization
def plot_elbow_method(matrix, method='kmeans', max_clusters=10, save_path=None):
    distortions = []
    K = range(1, max_clusters + 1)
    
    for k in K:
        if method == 'kmeans':
            model = KMeans(n_clusters=k, random_state=0).fit(matrix)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=k).fit(matrix)
        else:
            raise ValueError("Unsupported clustering method.")
        
        distortions.append(model.inertia_ if method == 'kmeans' else sum(model.children_))
    
    plt.figure(figsize=(10, 7))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method showing the optimal k')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def visualize_clusters(matrix, labels, method='pca', filenames=None, save_path=None):
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Unsupported reduction method.")
    
    reduced_matrix = reducer.fit_transform(matrix)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c=labels, cmap='viridis')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title("Cluster Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    if filenames:
        for i, filename in enumerate(filenames):
            plt.annotate(filename, (reduced_matrix[i, 0], reduced_matrix[i, 1]), fontsize=8, alpha=0.75)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


# we create a pdf report, includes all the visulization and the best method
def generate_cluster_report(filenames, labels, output_path, tfidf_elbow_plot_path=None, tfidf_cluster_plot_path=None, word2vec_elbow_plot_path=None, word2vec_cluster_plot_path=None):
    cluster_df = pd.DataFrame({'Filename': filenames, 'Cluster': labels})
    cluster_counts = cluster_df['Cluster'].value_counts().sort_index()
    cluster_terms = cluster_df.groupby('Cluster')['Filename'].apply(lambda x: ', '.join(x)).reset_index()

    report_text = f"Number of clusters: {len(cluster_counts)}\n\n"
    report_text += "Cluster Counts:\n"
    for cluster, count in cluster_counts.items():
        report_text += f"Cluster {cluster}: {count} papers\n"

    report_text += "\nCluster Terms:\n"
    for _, row in cluster_terms.iterrows():
        report_text += f"Cluster {row['Cluster']}: {row['Filename']}\n"

    # PDF generation
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.multi_cell(0, 10, report_text)
    
    # Add TF-IDF Elbow Plot to the pdf
    if tfidf_elbow_plot_path:
        pdf.add_page()
        pdf.image(tfidf_elbow_plot_path, x=10, y=10, w=pdf.w - 20)
    
    # Add TF-IDF Cluster Plot to the pdf
    if tfidf_cluster_plot_path:
        pdf.add_page()
        pdf.image(tfidf_cluster_plot_path, x=10, y=10, w=pdf.w - 20)
    
    # Add Word2Vec Elbow Plot to the pdf
    if word2vec_elbow_plot_path:
        pdf.add_page()
        pdf.image(word2vec_elbow_plot_path, x=10, y=10, w=pdf.w - 20)
    
    # Add Word2Vec Cluster Plot to the pdf
    if word2vec_cluster_plot_path:
        pdf.add_page()
        pdf.image(word2vec_cluster_plot_path, x=10, y=10, w=pdf.w - 20)
    
    pdf.output(output_path)
    print(f"Cluster report saved to {output_path}")

def main():
    text_directory = "./Abstract"
    report_path = "./Cluster_Report.pdf"
    tfidf_elbow_plot_path = "./TFIDF_Elbow_Method.png"
    tfidf_cluster_plot_path = "./TFIDF_Cluster_Visualization.png"
    word2vec_elbow_plot_path = "./Word2Vec_Elbow_Method.png"
    word2vec_cluster_plot_path = "./Word2Vec_Cluster_Visualization.png"
    
    texts, filenames = load_and_preprocess_texts(text_directory)
    
    if not texts:
        print("No texts to process.")
        return

    tfidf_matrix, tfidf_vectorizer = tfidf_vectorize(texts)
    best_tfidf_labels = None
    best_tfidf_n_clusters = None
    best_tfidf_silhouette_score = None
    best_tfidf_db_score = None
    if tfidf_matrix is not None:
        plot_elbow_method(tfidf_matrix.toarray(), method='kmeans', save_path=tfidf_elbow_plot_path)
        best_n_clusters, tfidf_silhouette_score, tfidf_db_score, tfidf_labels = evaluate_clustering(tfidf_matrix.toarray(), method='kmeans')
        if best_n_clusters == 2:  
            best_tfidf_labels = tfidf_labels
            best_tfidf_n_clusters = best_n_clusters
            best_tfidf_silhouette_score = tfidf_silhouette_score
            best_tfidf_db_score = tfidf_db_score
            print(f"TF-IDF -> Best number of clusters: {best_n_clusters}, Silhouette Score: {tfidf_silhouette_score}, Davies-Bouldin Score: {tfidf_db_score}")
            visualize_clusters(tfidf_matrix.toarray(), tfidf_labels, method='pca', filenames=filenames, save_path=tfidf_cluster_plot_path)
        else:
            print(f"TF-IDF -> Best number of clusters: {best_n_clusters}, Silhouette Score: {tfidf_silhouette_score}, Davies-Bouldin Score: {tfidf_db_score}")

    word2vec_matrix, word2vec_model = word2vec_vectorize(texts)
    best_word2vec_labels = None
    best_word2vec_n_clusters = None
    best_word2vec_silhouette_score = None
    best_word2vec_db_score = None
    if word2vec_matrix is not None:
        plot_elbow_method(word2vec_matrix, method='kmeans', save_path=word2vec_elbow_plot_path)
        best_n_clusters, word2vec_silhouette_score, word2vec_db_score, word2vec_labels = evaluate_clustering(word2vec_matrix, method='kmeans')
        if best_n_clusters == 2:  
            best_word2vec_labels = word2vec_labels
            best_word2vec_n_clusters = best_n_clusters
            best_word2vec_silhouette_score = word2vec_silhouette_score
            best_word2vec_db_score = word2vec_db_score
            print(f"Word2Vec -> Best number of clusters: {best_n_clusters}, Silhouette Score: {word2vec_silhouette_score}, Davies-Bouldin Score: {word2vec_db_score}")
            visualize_clusters(word2vec_matrix, word2vec_labels, method='pca', filenames=filenames, save_path=word2vec_cluster_plot_path)
        else:
            print(f"Word2Vec -> Best number of clusters: {best_n_clusters}, Silhouette Score: {word2vec_silhouette_score}, Davies-Bouldin Score: {word2vec_db_score}")

    
    generate_cluster_report(filenames, best_tfidf_labels if best_tfidf_n_clusters == 2 else best_word2vec_labels, report_path, tfidf_elbow_plot_path, tfidf_cluster_plot_path, word2vec_elbow_plot_path, word2vec_cluster_plot_path)

if __name__ == "__main__":
    main()
