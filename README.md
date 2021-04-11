# Humor-Detection

### Remaining To Do (no particular order)

1) Use `bert-base-uncased` to get baseline of original paper we are basing on.

2) Implement gradient accumulation steps + argument to determine number of steps to accumulate.
   
3) Implement scheduler for linear warmup to improve our optimizer (not that important yet).

4) Download data for other 2 tasks and implement dataset classes that inherit the same member functions as
our other dataset class.
   
5) Run experiments using ambiguity scoring on Reddit data