Calculates the 2-point correlation function and jackknife covariance given some 
data and randoms, and can also do some bao fitting and will make some plots as 
well as a not-best-fit model for reference.

For generating the correlation function there are 4 different scrips (just 
edit the batch script if you want to run them all) that cover the un-reconstructed 
and reconstructed correlation functions, as well as a version that calculates 
the jackknife covariance using a smaller sample size (to save calculation time)
for both the unreconstructed and reconstructed correlation functions.

NOTE: If you're not working with Roman mocks you'll likely have to rename
some of the columns from the data you're importing, as well as get rid of the
line confusion stuff (it'll be commented where you need to delete/edit a line).

baofitscript.py generates a model line to compare with the one calculated from
your data, and also gives you some BAO-fitting information.

zs.py just prints out some basic info about your galaxy input data, and also creates
a couple plots to visualize it.

BAOfit.py has all of the code to generate the model line but you shouldn't need 
to edit it directly too much.

plots.py does what you might expect it to do and makes a bunch of plots based
on the correlation functions you've generated (check to make sure you've made all
the ones it needs).

Conda environment under package-list.txt

Happy correlating!