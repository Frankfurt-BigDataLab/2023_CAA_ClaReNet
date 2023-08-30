# Die Study
----

Notebooks:
- `Preprocessing`: Use this to apply the filters used for preprocessing the image.
- `Extract_Matches`: Use this to extract the matches in a directory and to save them in a csv file.

After extracting the matches we used [Orange Data Mining](https://orangedatamining.com/) to apply the hierarchical clustering and visualise the result.

<img src="figures/orange1.jpg"  width="400" height="200"> <img src="figures/orange_gif.gif"  width="400" height="200">

----
# Requirements
- imageio or scipy <= 1.1
- joblib
- matplotlib
- numpy
- opencv
- pandas

----
You can import the workflow we created in [Orange Data Mining](https://orangedatamining.com/). You can find it in the `Orange` folder.  
The input for the workflow is a matrix between all the images as a csv file, as created by `Extract_Matches`. 
To visualise the images, a path to the files is necessary.
