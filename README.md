run with python3 acfpcf.py --lag 672 672 is changeable, keep in mind its 15min data
lag is how far back in time you want to look for correlations
lag of 4 compares each return with the return 4 periods back
lag of 16 compares each return with the return 16 periods back
when we set --lag 96, use all of our data(every 15 minute return we have)
for each point, calculate correlation with up to 96 periods back
this will give us a sense of how returns are correlated over time
