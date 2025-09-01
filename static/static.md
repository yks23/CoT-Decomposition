# GSM8K @1
llama3.1-instruct-with-zeroshot 0.8316906747536013
mistral-instruct-with-zeroshot 0.5572403335860501
llama3.1-instruct-without-zeroshot 0.844579226686884
mistral-instruct-without-zeroshot 0.6110689916603488
# MATH @1
llama3.1-instruct-with-zeroshot 60.99
mistral-instruct-with-zeroshot 27.50
llama3.1-instruct-without-zeroshot 63.69
mistral-instruct-without-zeroshot 28.05

gsm8k
temp 0 prompt @5  0.8097043214556482 vs 0.8097043214556482
temp 1 prompt @5  94.24 vs 0.7821076573161486
temp mixed prompt @5 onprocessing
temp 0 no prompt @5 onprocessing 0.844579226686884 vs 0.844579226686884
temp 1 no prompt @5 onprocessing

math

1. prompt工程会影响效果，没办法不训练
2. prompt+temperature实验+@topk实验

temp 1 with prompt
1 2 3 4 5
0.78468 0.8817 0.9151 0.9303 0.9424
80.82 85.29 86.96 88.57