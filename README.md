# ndsb_theano
Python/Theano code used for the NDSB Kaggle Competition (Convolutional Neural Network)

This is the code I used for my submission(s) to the Kaggle NDSB competition on image processing and classification of 
ocean plankton.

I'll be honest, this is not the cleanest code I've ever written. It started with code you can see from this fantastic YouTube
video: https://www.youtube.com/watch?v=S75EdAcXHKk  (search for "Deep Learning Theano" and it will be the first result)

The tutorial material on http://www.deeplearning.net/tutorial/ is great, and I learnt a lot from it, but the code in those 
tutorials is rather large, making it not always easy to fully understand what's going on. The code in the YT vid is much sparser
which I found easier to understand.

That's not to say I didn't bastardize this. There's plenty of spaghetti in there that should have been turned into functions,
but what can you do. The code grew as I went along, and much of the additional code is really to get more data out of the 
network to see what it was doing. (including plotting the node weights as images, as well as plotting some of the validation 
set performance.

Make sure you have Theano working, including NumPy, SciPy and such, before getting started, and if you have a GPU at your disposal 
set that up first. These instructions + links in it worked for me: http://deeplearning.net/software/theano/install.html#install 

This does take quite a while, and in my case, a day (but included a clean Linux install as well). You'll want to run 
on GPU if you can, because it proved easily something like 20-50 times faster. For me this also involved compiling and installing 
an Nvidia kernel driver, so is not exactly trivial. NumPy and SciPy install relatively quickly, but Theano will run about an hour+
worth of tests. In the end, it's worth it, though. I ran first on another Linux box without GPU, running my first model while 
I prepped the new box. With GPU (a GTX Titan) I ran in 20 minutes (!) what CPU had taken 24 hours over... (Considering I actually
ran a 5 convolutional layer model at one point (model3 in the code) that took around 24 hrs on GPU, you can imagine what that 
would have been on CPU....)

As I said, the code is a little ugly, and has parts cobbled together from elsewhere, as well as my "diagnostics" code to 
understand what the model was doing. Perhaps the only redeeeming part is that I often used the conv_net.py file to keep notes,
but these are by no means complete.

To run this code, make these folders:
- filters
- test_img
- saved_params
- plots
   
In filters you'll get the state of the layer weights/filters at the end of each training epoch. Test_img creates a .tiff image
during the load of the training data once for each folder/category, so you can get a sense what the network operates on. 
Initially I simply resized/scaled the images to a square format - in the current code I first add white space and only after that
perform the scaling. (There's also a small gamma adjustment to enhance contrast). In saved_params the weights.pkl are saved at 
the end of every validation test at the end of an epoch. That means that potentially you could ^C at any point, and run full predictions
with the last saved layer weights. Note that in that case you should NOT train again, because you'll likely overfit as it 
won't create the exact same training and validation set... So, only do this for predictions, not for training.

Finally, in plots we save the first 12 results of the validation test vs the known labels. I added this in to see how the 
network learns over time.

To increase the training/validation set, I rotated all original source images by 90 degrees. The code for that is in rotate_training_set.py.
Run this _before_ running the conv_net.py itself. Note that load_images.py loads the image files, both the training and validation
sets (including randomization and split) and the actual test set. utils.py comes from here: http://www.deeplearning.net/tutorial/utilities.html

The best result I got was a validation score of around 1.121 or so (roughly equivalent to ~65% accuracy rate), which is well 
off from the leaders in the NDSB, but perhaps this code is still useful for some. The exercise certainly was very educational to me.
