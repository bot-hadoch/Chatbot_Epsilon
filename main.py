import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report

def preprocess_pandas(data, columns):
    df_ = pd.DataFrame(columns=columns)
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
    data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
    data['Sentence'] = data['Sentence'].str.replace('[^\w\s]','')                                                       # remove special characters
    data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)                                                   # remove numbers
    for index, row in data.iterrows():
        word_tokens = word_tokenize(row['Sentence'])
        filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
        df_ = df_.append({
            "index": row['index'],
            "Class": row['Class'],
            "Sentence": " ".join(filtered_sent[0:])
        }, ignore_index=True)
    return data

def train(x_data, y_data, vocabulary_size, epochs):
    
    
    
    network = nn.Sequential(
        nn.Linear(vocabulary_size, 10),
        nn.ReLU(),
        nn.Linear(10, 3)        
    )
    
    loss_function = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(network.parameters())
    
    for epoch in range(epochs):

        prediction = network(x_data)               
            
        loss = loss_function(prediction, y_data)
            
        loss.backward()
            
        optimizer.step()
            
        optimizer.zero_grad()
            
        if(epoch %10 == 0):
            print("epoch: ")
            print(epoch)
            print("loss: ")
            print(loss.item())
        
        
    return network

def validate(network, x_data, y_data):
    total = 0
    correct = 0
    prediction = network(x_data)
    print("validating...")
    for i in range(len(prediction)):
        total += 1
        guess = torch.argmax(prediction[i])
        if guess.item() == y_data[i].item():
            correct += 1
    print("total:")
    print(total)
    print("correct:")
    print(correct)
    
    
def review_product():
    review = input("What did you think about the product?\n")
    
    review_list = []
    
    review_list.append(review)
    
    test_data = word_vectorizer.transform(review_list)
    test_data = test_data.todense()
    test_data_x_tensor = torch.from_numpy(np.array(test_data)).type(torch.FloatTensor)
    test_prediction = network(test_data_x_tensor)
    if torch.argmax(test_prediction[0]) == 0:
        print("I am sorry to hear you did not like the product.")
    elif torch.argmax(test_prediction[0]) == 1:
        print("I am happy to hear you liked the product.")
    else:
        print("hello")
    

# If this is the primary file that is executed (ie not an import of another file)
if __name__ == "__main__":
    # get data, pre-process and split
    data = pd.read_csv("amazon_cells_labelled2.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index                                          # add new column index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)                             # pre-process
    training_data, validation_data, training_labels, validation_labels = train_test_split( # split the data into training, validation, and test splits
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.10,
        random_state=0,
        shuffle=True
    )

    # vectorize data using TFIDF and transform for PyTorch for scalability
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000, max_df=0.5, use_idf=True, norm='l2')
    training_data = word_vectorizer.fit_transform(training_data)        # transform texts to sparse matrix
    training_data = training_data.todense()                             # convert to dense matrix for Pytorch
    vocab_size = len(word_vectorizer.vocabulary_)
    validation_data = word_vectorizer.transform(validation_data)
    validation_data = validation_data.todense()
    train_x_tensor = torch.from_numpy(np.array(training_data)).type(torch.FloatTensor)
    train_y_tensor = torch.from_numpy(np.array(training_labels)).long()
    validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
    validation_y_tensor = torch.from_numpy(np.array(validation_labels)).long()
    
    
    network = train(train_x_tensor, train_y_tensor, vocab_size, 1000)
    print("done training")
    validate(network, validation_x_tensor, validation_y_tensor)
    
    while True:
        review_product()
    
    
    
