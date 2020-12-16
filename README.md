# CS-GY-9233-Deep-Learning, New York University
Question Answering with Contextual Embeddings, Data Augmentation, and Bi-directional Attentional Flow on SQuAD 2.0 Dataset.
# What is SQuAD 2.0?
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. This project aims to solve the task of Question Answering from a context paragraph using Ensemble of state of the art NLP models like RoBERTa, BERT, ALBERT, DistilBERT, and BiDAF. 

# Requirements
All the notebooks can be executed directly on Google Colab, to run these files on a local machine, one will need
1. torch
2. pytorch-transformers
3. Scientific computing packages - numpy, pandas
4. NLTK, Wordnet

# Description of each file
1. bidaf_modified.ipynb contains outputs obtained after modifying BiDAF with BERT embeddings.
2. bidaf_with_one_full_pass.ipynb contains one full pass performed on a BiDAF network with BERT embeddings.
3. embedding_replacement.ipynb contains the code for replacing GLOVE embeddings with BERT embeddings.
4. ensemble_with_bidaf.ipynb contains ensembling procedure using ALBERT, BERT, DistilBERT, RoBERTa, and BiDAF.
5. ensemble_without_bidaf.ipynb contains ensembling procedure using ALBERT, BERT, DistilBERT, RoBERTa. 
6. Synonym Replacement.ipynb contains code for performing synonym replacement and adding the modified data to the original dataset.
7. gpt_2_context_extension.ipynb contains code for performing context extension using GPT-2

# Results
![alt text](https://github.com/RishikeshDhayarkar/CS-GY-9233-Deep-Learning/blob/main/Comparison.png)
![alt text](https://github.com/RishikeshDhayarkar/CS-GY-9233-Deep-Learning/blob/main/Results.png)

# Conclusion
In  this  project  we  were  able  to  implement  a  robust  QA  system  that  achieved  a  top  EM/F1  of 81.13/84.06,  which is an increment of +1.34/2.17on EM/F1.  To achieve this boost in per-formance we first performed data augmentation via synonym replacement and context extensionusing GPT-2.  Then, we explored the BiDAF architecture and improved its baseline performanceby an EM/F1 of +3.51/4.04.  This was done by replaceing GLOVE embeddings in the baselineBiDAF  by  BERT  embeddings.   We  expected  a  classical  QA  model  like  BiDAF  to  augment  theperformance of a PCE models in the form of a robust ensemble, but this was not the case.  The en-semble without BiDAF performed better than the one with BiDAF. The ensemble without BiDAFbeats the ensemble with BiDAF by an EM/F1 of 0.78/0.32. 

By replacing the ’base’ versions by ’xlarge’ and’xxlarge’ versions of the models we expect a significant boost in the EM and F1 scores.  For example, from our literature survey we found out that the base version of ALBERT gives an EM/F1 of 78.63/81.11whereas, the ’xxlarge version gives an EM/F1 of 86.3/88.9.  By implementing themethods from this project on ’xlarge’ and ’xxlarge’ models, the final scores of the ensembles candefinitely beat human performance on SQuAD 2.0.
