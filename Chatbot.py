#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np


# In[2]:


with open("train_qa.txt","rb") as fp:
  train_data=pickle.load(fp)


# In[3]:


train_data


# In[4]:


with open("test_qa.txt","rb") as fp:
  test_data=pickle.load(fp)


# In[5]:


test_data


# In[6]:


type(train_data)


# In[7]:


type(test_data)


# In[8]:


len(train_data)


# In[9]:


len(test_data)


# In[10]:


train_data[0]


# In[11]:


' '.join(train_data[0][0])


# In[12]:


train_data[0][1]


# In[13]:


' '.join(train_data[0][1])


# In[14]:


train_data[0][2]


# In[15]:


' '.join(train_data[0][2])


# In[16]:


#Set up vocabulary
vocab=set()


# In[17]:


all_data=test_data+train_data


# In[18]:


type(all_data)


# In[19]:


all_data


# In[20]:


for a in all_data:
    print(a)
    break


# In[21]:


for story,question,answer in all_data:
    vocab=vocab.union(set(story))
    vocab=vocab.union(set(question))


# In[22]:


vocab.add('yes')
vocab.add('no')


# In[23]:


vocab


# In[24]:


len(vocab)


# In[25]:


vocab_len=len(vocab)+1


# In[26]:


max_story_len=max([len(data[0]) for data in all_data])
max_story_len


# In[27]:


max_ques_len=max([len(data[1]) for data in all_data])
max_ques_len


# #Vectorizer

# In[28]:


vocab


# In[29]:


tf.keras.preprocessing.sequence.pad_sequences 
tf.keras.preprocessing.text.Tokenizer


# In[ ]:


tokenizer=Tokenizer(filters=[])


# In[ ]:


tokenizer.fit_on_texts(vocab)


# In[ ]:


tokenizer.word.index


# In[ ]:


train_story_seq


# In[ ]:


train_story_text


# In[ ]:


def vectorize_stories(data,word_index=tokenizer.word_index,max_story_len=max_story_len,max_ques_len=max_ques_len):
    X=[]#stories
    Xq=[] #query/question
    Y=[] #correct answer
    
    for story,quer,answer in data:
        x=[word_index[word.lower()]for word in story]
        xq=[word_index[word.lower()]for word in query]
        y=np.zeros(len(word_index)+1)
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        
       return(pad_sequence(X,max_len=max_story_len),pad_sequence(Xq,max_len=max_ques_len),np.array(Y)) 
        
        
    


# In[ ]:


input_train,query_train,answer_train=vectorize_stories(train_data)
input_test,query_test,answer_test=vectorize_stories(test_data)


# In[ ]:


input_train


# In[ ]:


query_test


# In[ ]:


answer_test


# In[ ]:


tokenizer.word_index('yes')


# In[ ]:


tokenizer.word_index('no')


# # Creating Model

# In[ ]:


from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers import Input,Activation,Dense,Permute,Dropout,add,dot,concatenate,LSTM


# In[ ]:


input_sequence=Input((max_story_len),)
question=Input((max_ques_len),)


# In[ ]:


#Input encoder m
input_encoder_m=Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_len,output_dim=64))
input_encoder_m.add(Dropout(0.3))


# In[ ]:


#Input encoder c
input_encoder_c=Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_len,output_dim=max_ques_len))
input_encoder_c.add(Dropout(0.3))


# In[ ]:


#Question encoder
question_encoder=Sequential()
question_encoder.add(Embedding(input_dim=vocab_len,output_dim=64,input_length=max_ques_len))
question_encoder.add(Dropout(0.3))


# In[ ]:


#Encode the sequence
input_encoder_m=input_encoder_m(input_sequence)

input_encoder_c=input_encoder_c(input_sequence)
question_encoder=question_encoder(question)


# In[ ]:


match=dot([input_encoded_m,question_encoded],axes=(2,2))
match=Activation('softmax')(match)


# In[ ]:


response=add([match,input_encoded_c])
response=Permute((2,1))(response)


# In[ ]:


#Concatenate
answer=concatenate([response,question_encoded])


# In[ ]:


answer


# In[ ]:


answer.LSTM(32)(answer)


# In[ ]:


answer=Dropout(0.5)(answer)


# In[ ]:


answer=Dense(vocab_len)(answer)


# In[ ]:


answer=Activation('softmax')(answer)


# In[ ]:


model=Model([input_sequence,question],answer)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history=model.fit([input_train,queries_train],answer_train,batch_size=32,epochs=5,
                  validation_data=([input_test,queries_test],answer_test))


# In[ ]:


import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')


# In[ ]:




