# Paper-Abstraction-NLP

## Introduction about the project 
This project aims to develop a script that clusters a given set of research papers based on their abstract similarity.

## Key words:
Natural Language processing, Text processing, Unsupervised machine learning, Clustering, Visualisation

## Set up the environment
All the dependencies and library needs to be prepared. I recommend take advantage of virtual environment. 
One example for creating the virtual environment is using "virtualenv".

## Virtual Env
Active the bash environment and run the following command:
python3 -m venv venv
source venv/bin/activate



# Dependencies and Libraries
pip install -r requirements.txt 

Please note that the "requirements.txt" needs to be prepared manually, more detailed information will be given.

## Make sure the directory tree is the same as below


.
├── Abstracts                  # All the pdf files being downloaded
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...


 ├── script.py                  # script file

 ├── requirements.txt           # dependencies

 └── README.md                  # README file

# Run the script
\`\`\`bash
python script.py
\`\`\`


# More detail about the project script and functionalities.

## Pre-processing
The pre-processing process include downloading the paper in pdf format. 

## After running, the script doing the following things: 
1. Read the pdf file and fetch the "Abstract" part of the paper
2. Handling the content, removing stop words, stemming or lemmatizing words, and handling any special characters or formatting issues.
3. Vectorisation the text. This involves the vectorisation process using TF-IDF and word2Vector.
4. Evaluate the Clustering.
5. Visulation the Clustering results.
6. Generate a report and save the clustering results to a new file.


## About the `requirements.txt`

nltk==3.6.2

numpy==1.20.3

pandas==1.2.4

matplotlib==3.4.2

scikit-learn==0.24.2

gensim==4.0.1

PyPDF2==1.26.0



## About the text vectorisation
Basically, the vectorisation process involves the following techniques:     TF-IDF: Converts texts into a sparse matrix of TF-IDF scores for terms in the documents.
Word2Vec: Trains a Word2Vec model on tokenized texts and converts each text into a vector. This basically include averaging the word vectors.

## About the Evaluation:
The script applied Elbow Method and Silhouette Analysis to determine the optimal number of clusters. Also, the quality of the clustering results is being determined. 

## About the visualisation
Applied PCA , t-SNE to reduce the #dim. In the diagrams, the numebr of each paper are labelled. 

## Generate a Report
A well-structured reprot is generated after the program is executed.
The pdf report contains the result of the clustering and visualisation. 


