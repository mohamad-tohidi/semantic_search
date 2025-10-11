
we need a good sample of the index

the index has 240 thousand data in it

lets just randomly pick 10.000 of them



how do we save the sample?

i think as a jsonl file
with a pydantic model for it would be nice


ok, i have gotten the data

now we need to do what?
we have to setup a simple qdrant client

load the model that we want to use (e5-large)
then chunk up the data and save them in the qdrant