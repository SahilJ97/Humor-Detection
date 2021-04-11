# Humor-Detection

### Remaining To Do

1) Use `bert-base-uncased` to get baseline of original paper we are basing on. Should really be as
simple as altering the line that loads in our model and running the training time.
   
2) Implement scheduler for linear warmup to improve our optimizer (not that important yet).

3) Download data for other 2 tasks and implement dataset classes that inherit the same member functions as
our other dataset class.
   
4) Run experiments using ambiguity scoring on Reddit data 