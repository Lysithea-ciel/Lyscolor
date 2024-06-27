# Lyscolor
Lysithea_ciel's color processing software, intended to aid artists in analyzing color compositions and in picking colors.
This program takes an input image and processes it, largely to apply filters and graph the colors used in the OKLab color space.
The input image must be a .png or .jpg, located in the same folder that the program is. It picks the first image it finds, and the output images are put into the same folder- this means that you'll probably have to move the output images out of the folder before running it again, or it'll start processing one of those.

The image variants that the program will create are:  
-Perceptual grayscale (i.e., isolated perceived brightness)  
-Isolated red-green spectrum  
-Isolated blue-yellow spectrum  
-Isolated hue angle  
-Isolated chroma

It will also create a suggested gradient- this is done by averaging the colors used in the image (well, a selection of them) into a plane, and interpreting that plane as a slice of the OKLab color space. Color picking from this plane, or using colors close to it in your art, may improve your colors.

Finally, the program will graph the colors used in the image in OKLab- this can be surprisingly informative. You can choose whether or not you want to put the suggested gradient in that same graph as well.
