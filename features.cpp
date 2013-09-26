#include <assert.h>
#include <math.h>
#include <FL/Fl.H>
#include <FL/Fl_Image.H>
#include "features.h"
#include "ImageLib/FileIO.h"

#define PI 3.14159265358979323846
#define E  2.71828182845904523536

// Compute features of an image.
bool computeFeatures(CFloatImage &image, FeatureSet &features, int featureType, int descriptorType) {
    // TODO: Instead of calling dummyComputeFeatures, implement
    // a Harris corner detector along with a MOPS descriptor.  
    // This step fills in "features" with information necessary 
    // for descriptor computation.

    switch (featureType) {
    case 1:
        dummyComputeFeatures(image, features);
        break;
    case 2:
        ComputeHarrisFeaturesScale(image, features);
        break;
    default:
        return false;
    }

    // TODO: You will implement two descriptors for this project
    // (see webpage).  This step fills in "features" with
    // descriptors.  The third "custom" descriptor is extra credit.
    switch (descriptorType) {
    case 1:
        ComputeSimpleDescriptors(image, features);
        break;
    case 2:
        ComputeMOPSDescriptors(image, features);
        break;
    case 3:
        ComputeCustomDescriptors(image, features);
        break;
    default:
        return false;
    }

    // This is just to make sure the IDs are assigned in order, because
    // the ID gets used to index into the feature array.
    for (unsigned int i=0; i<features.size(); i++) {
        features[i].id = i;
    }

    return true;
}

// Perform a query on the database.  This simply runs matchFeatures on
// each image in the database, and returns the feature set of the best
// matching image.
bool performQuery(const FeatureSet &f, const ImageDatabase &db, int &bestIndex, vector<FeatureMatch> &bestMatches, double &bestDistance, int matchType) {
    vector<FeatureMatch> tempMatches;

    for (unsigned int i=0; i<db.size(); i++) {
        if (!matchFeatures(f, db[i].features, tempMatches, matchType)) {
            return false;
        }

        bestIndex = i;
        bestMatches = tempMatches;
    }

    return true;
}

// Match one feature set with another.
bool matchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches, int matchType) {

    // TODO: We have provided you the SSD matching function; you must write your own
    // feature matching function using the ratio test.

    printf("\nMatching features.......\n");

    switch (matchType) {
    case 1:
        ssdMatchFeatures(f1, f2, matches);
        return true;
    case 2:
        ratioMatchFeatures(f1, f2, matches);
        return true;
	case 3:
		improvedratioMatchFeatures(f1, f2, matches);
		return true;
    default:
        return false;
    }
}

// Compute silly example features.  This doesn't do anything
// meaningful, but may be useful to use as an example.
void dummyComputeFeatures(CFloatImage &image, FeatureSet &features) {
    CShape sh = image.Shape();
    Feature f;

    for (int y=0; y<sh.height; y++) {
        for (int x=0; x<sh.width; x++) {
            double r = image.Pixel(x,y,0);
            double g = image.Pixel(x,y,1);
            double b = image.Pixel(x,y,2);

            if ((int)(255*(r+g+b)+0.5) % 100 == 1) {
                // If the pixel satisfies this meaningless criterion,
                // make it a feature.

                f.type = 1;
                f.id += 1;
                f.x = x;
                f.y = y;

                f.data.resize(1);
                f.data[0] = r + g + b;

                features.push_back(f);
            }
        }
    }
}

// Compute features for 3 level scaled images (Implemented to apply
// scale invariance of the feature detection)
void ComputeHarrisFeaturesScale(CFloatImage &image, FeatureSet &features)
{
	// Feature set for the 5 scale levels
	FeatureSet features1;
	FeatureSet features2;
	FeatureSet features3;
	FeatureSet features4;
	FeatureSet features5;
	
	int w = image.Shape().width;
	int h = image.Shape().height;

	CFloatImage oriFiltered(w, h, 3);
	CFloatImage halfFiltered(w/2, h/2, 3);
	CFloatImage scale_half(w/2, h/2, 3);
	CFloatImage scale_onethird(w/3, h/3, 3);
	CFloatImage scale_onefourth(w/4, h/4, 3);
	CFloatImage scale_onefifth(w/5, h/5, 3);

	CFloatImage LPFilter(5, 5, 1);
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
			LPFilter.Pixel(i, j, 1) = gaussian5x5[5 * j + i];
	// Downsaple to 1/2
	Convolve(image, oriFiltered, LPFilter);
	for (int i = 0; i < scale_half.Shape().width; i++)
		for (int j = 0; j < scale_half.Shape().height; j++)
			scale_half.Pixel(i, j, 1) = oriFiltered.Pixel(i * 2, j * 2, 1);
	// Downsample to 1/3
	for (int i = 0; i < scale_onethird.Shape().width; i++)
		for (int j = 0; j < scale_onethird.Shape().height; j++)
			scale_onethird.Pixel(i, j, 1) = oriFiltered.Pixel(i * 3, j * 3, 1);
	// Downsample to 1/4
	Convolve(scale_half, halfFiltered, LPFilter);
	for (int i = 0; i < scale_onefourth.Shape().width; i++)
		for (int j = 0; j < scale_onefourth.Shape().height; j++)
			scale_onefourth.Pixel(i, j, 1) = halfFiltered.Pixel(i * 2, j * 2, 1);
	// Downsample to 1/5
	for (int i = 0; i < scale_onefifth.Shape().width; i++)
		for (int j = 0; j < scale_onefifth.Shape().height; j++)
			scale_onefifth.Pixel(i, j, 1) = halfFiltered.Pixel(i * 2.5, j * 2.5, 1);
			
	// Compute features for different scales
	ComputeHarrisFeatures(image, features1);
	ComputeHarrisFeatures(scale_half, features2);
	ComputeHarrisFeatures(scale_onethird, features3);
	ComputeHarrisFeatures(scale_onefourth, features4);
	ComputeHarrisFeatures(scale_onefifth, features5);

	// Upsample and merge the features
	for (vector<Feature>::iterator i = features1.begin(); i != features1.end(); i++)
	{
		Feature &f1 = *i;
		features.push_back(f1);
	}
	for (vector<Feature>::iterator j = features2.begin(); j != features2.end(); j++)
	{
		Feature &f2 = *j;
		f2.x *= 2;
		f2.y *= 2;
		if (image.Shape().InBounds(f2.x, f2.y))
			features.push_back(f2);
	}
	for (vector<Feature>::iterator k = features3.begin(); k != features3.end(); k++)
	{
		Feature &f3 = *k;
		f3.x *= 3;
		f3.y *= 3;
		if (image.Shape().InBounds(f3.x, f3.y))
			features.push_back(f3);
	}
	for (vector<Feature>::iterator m = features4.begin(); m != features4.end(); m++)
	{
		Feature &f4 = *m;
		f4.x *= 4;
		f4.y *= 4;
		if (image.Shape().InBounds(f4.x, f4.y))
			features.push_back(f4);
	}
	for (vector<Feature>::iterator n = features5.begin(); n != features5.end(); n++)
	{
		Feature &f5 = *n;
		f5.x *= 5;
		f5.y *= 5;
		if (image.Shape().InBounds(f5.x, f5.y))
			features.push_back(f5);
	}
	//printf("Features Merged!\n");
}

void ComputeHarrisFeatures(CFloatImage &image, FeatureSet &features)
{
    //Create grayscale image used for Harris detection
    CFloatImage grayImage = ConvertToGray(image);
	
    //Create image to store Harris values
    CFloatImage harrisImage(image.Shape().width,image.Shape().height,1);

    //Create image to store local maximum harris values as 1, other pixels 0
    CByteImage harrisMaxImage(image.Shape().width,image.Shape().height,1);

    CFloatImage orientationImage(image.Shape().width, image.Shape().height, 1);

    // computeHarrisValues() computes the harris score at each pixel position, storing the
    // result in in harrisImage. 
    // You'll need to implement this function.

    computeHarrisValues(grayImage, harrisImage, orientationImage);

    // Threshold the harris image and compute local maxima.  You'll need to implement this function.
    computeLocalMaxima(harrisImage,harrisMaxImage);

    CByteImage tmp(harrisImage.Shape());
    convertToByteImage(harrisImage, tmp);
    WriteFile(tmp, "harris.tga");
    // WriteFile(harrisMaxImage, "harrisMax.tga");

    // Loop through feature points in harrisMaxImage and fill in information needed for 
    // descriptor computation for each point above a threshold. You need to fill in id, type, 
    // x, y, and angle.
    int id = 0;
    for (int y=0; y < harrisMaxImage.Shape().height; y++) {
        for (int x=0; x < harrisMaxImage.Shape().width; x++) {

            if (harrisMaxImage.Pixel(x, y, 0) == 0)
                continue;

            Feature f;

            //TODO: Fill in feature with location and orientation data here 
			f.type=2;
			f.id=id;
            f.x = x;
            f.y = y;
			f.angleRadians=orientationImage.Pixel(x,y,0);

            features.push_back(f);
            id++;
        }
    }
	//printf("ComputeHarrisFeatures Done!\n");
}



//TO DO---------------------------------------------------------------------
// Loop through the image to compute the harris corner values as described in class
// srcImage:  grayscale of original image
// harrisImage:  populate the harris values per pixel in this image
void computeHarrisValues(CFloatImage &srcImage, CFloatImage &harrisImage, CFloatImage &orientationImage)
{
    int w = srcImage.Shape().width;
    int h = srcImage.Shape().height;

    // TODO: You may need to compute a few filtered images to start with
	CFloatImage deltax(w,h,1);
	CFloatImage deltay(w,h,1);
	CFloatImage mask(5,5,1);
	
	//compute derivative images
	for (int y=0;y<h;y++)
	{
		for (int x=0;x<w;x++)
		{
			if ((x>0)&&(x<(w-1))&&(y>0)&&(y<(h-1)))
			{
				deltax.Pixel(x,y,0)=(srcImage.Pixel(x+1,y,0)-srcImage.Pixel(x-1,y,0))*255/2;
				deltay.Pixel(x,y,0)=(srcImage.Pixel(x,y+1,0)-srcImage.Pixel(x,y-1,0))*255/2;
			}
			//if on the boundary, no one would like to use it as a feature, so simply set 0
			else
			{
				deltax.Pixel(x,y,0)=0;
				deltay.Pixel(x,y,0)=0;
			}
		}
	}

	//compute the mask, using sigma=1
	double sumofmask;
	double distance;
	//IMPORTANT PARAMETER HERE: sigma2 of the gaussian weight kernel
	double sigma2=2;
	sumofmask=0;
	for (int y=0;y<5;y++)
	{
		for (int x=0;x<5;x++)
		{
			distance=((y-2)*(y-2)+(x-2)*(x-2));
			mask.Pixel(x,y,0)=pow(E,-distance/(2*sigma2));
			sumofmask+=mask.Pixel(x,y,0);
		}
	}
	//normalize the mask, though not necessary
	for (int y=0;y<5;y++)
	{
		for (int x=0;x<5;x++)
		{
			mask.Pixel(x,y,0)=mask.Pixel(x,y,0)/sumofmask;
		}
	}
	
	//printf("computeHarrisValues: Pre-filter Done!\n"); 

    double h1,h2,h3,h4;
	double lamda;
	double epsilon=0.001;
	for (int y = 0; y < h; y++) 
	{
        for (int x = 0; x < w; x++) 
		{

            // TODO:  Compute the harris score for 'srcImage' at this pixel and store in 'harrisImage'.  See the project
            //   page for pointers on how to do this.  You should also store an orientation for each pixel in 
            //   'orientationImage'
			h1=0;h2=0;h3=0;h4=0;
			for (int i=y-2;i<y+3;i++)
			{
				for (int j=x-2;j<x+3;j++)
				{
					if ((i>=0)&&(i<h)&&(j>=0)&&(j<w))
					{
						h1+=deltax.Pixel(j,i,0)*deltax.Pixel(j,i,0)*mask.Pixel(j-x+2,i-y+2,0);
						h2+=deltax.Pixel(j,i,0)*deltay.Pixel(j,i,0)*mask.Pixel(j-x+2,i-y+2,0);
						h3+=deltay.Pixel(j,i,0)*deltax.Pixel(j,i,0)*mask.Pixel(j-x+2,i-y+2,0);
						h4+=deltay.Pixel(j,i,0)*deltay.Pixel(j,i,0)*mask.Pixel(j-x+2,i-y+2,0);
					}
				}
			}
			harrisImage.Pixel(x,y,0)=(h1*h4-h2*h3)/(h1+h4+epsilon);
			lamda=((h1+h4)+sqrt(4*h2*h3+(h1-h4)*(h1-h4)))/2;
			orientationImage.Pixel(x,y,0) = (deltax.Pixel(x,y,0) > 0) ? atan((lamda-h1)/h2) : atan((lamda-h1)/h2) + PI;
			
        }
    }
	//printf("computeHarrisValues Done!\n"); 
}



//TO DO---------------------------------------------------------------------
//Loop through the image to compute the harris corner values as described in class
// srcImage:  image with Harris values
// destImage: Assign 1 to local maximum in 3x3 window, 0 otherwise
void computeLocalMaxima(CFloatImage &srcImage,CByteImage &destImage)
{
	int w = srcImage.Shape().width;
    int h = srcImage.Shape().height;

	//IMPORTANT PARAMETER HERE: threshold of the harris score
	double threshold=10;
	int ismax;
	int totalmax=0;
	for (int y=0;y<h;y++)
	{
		for (int x=0;x<w;x++)
		{
			if (srcImage.Pixel(x,y,0)<threshold)
			{
				ismax=0;
			}
			else
			{
				ismax=1;
				for (int i=y-2;i<y+3;i++)
				{
					for (int j=x-2;j<x+3;j++)
					{
						if ((i>=0)&&(i<h)&&(j>=0)&&(j<w)&&(i!=y)&&(j!=x))
						{
							if (srcImage.Pixel(j,i,0)>=srcImage.Pixel(x,y,0))
							{
								ismax=0;
							}
						}
					}
				}
			}
			if (ismax==1)
			{
				destImage.Pixel(x,y,0)=1;
				totalmax++;
			}
			else
			{
				destImage.Pixel(x,y,0)=0;
			}
		}
	}
	//printf("computeLocalMaxima Done!\n");

	//non-maximum suppression here
	FeatureSet tmp;
	int id = 0;
    for (int y=0; y < destImage.Shape().height; y++) 
	{
        for (int x=0; x < destImage.Shape().width; x++) 
		{
            if (destImage.Pixel(x, y, 0) == 0)
                continue;
            Feature f;
			//type indicates selection
			f.type=0;
			f.id=id;
            f.x = x;
            f.y = y;
			//this is f(x) stored in angleRadians, just using the data structure here
			f.angleRadians=srcImage.Pixel(x,y,0);
			//this indicates the radius
			f.data.resize(1);
			f.data[0]=1000000.0;
            tmp.push_back(f);
            id++;
        }
    }
	for (vector<Feature>::iterator i = tmp.begin(); i != tmp.end(); i++)
	{
		Feature &f1=*i;
		for (vector<Feature>::iterator j = tmp.begin(); j != tmp.end(); j++)
		{
			Feature &f2=*j;
			if (f1.angleRadians<(0.9*f2.angleRadians))
			{
				double distance=sqrt(((double)f1.x-(double)f2.x)*((double)f1.x-(double)f2.x)+((double)f1.y-(double)f2.y)*((double)f1.y-(double)f2.y));
				if (distance<f1.data[0])
				{
					f1.data[0]=distance;
				}
			}
		}
	}
	//IMPORTANT PARAMETER HERE: number of feature points
	int feature_num = (id < 500) ? id : 500;
	for (int y=0; y < destImage.Shape().height; y++) 
	{
        for (int x=0; x < destImage.Shape().width; x++) 
		{
			destImage.Pixel(x,y,0)=0;
		}
	}
	for (int i=0;i<feature_num;i++)
	{
		//find the max
		double max_rad=0;
		int nowwhich=0;
		int nowmax;
		vector<Feature>::iterator jjj;
		for (vector<Feature>::iterator j = tmp.begin(); j != tmp.end(); j++)
		{
			Feature &f3=*j;
			if ((f3.data[0]>max_rad)&&(f3.type==0))
			{
				max_rad=f3.data[0];
				nowmax=nowwhich;
				jjj=j;
			}
			nowwhich++;
		}
		//get to the max
		Feature &f4=*jjj;
		f4.type=1;
		destImage.Pixel(f4.x,f4.y,0)=1;
	}
	
}

// TODO: Implement parts of this function
// Compute Simple descriptors.
void ComputeSimpleDescriptors(CFloatImage &image, FeatureSet &features)
{
    //Create grayscale image used for Harris detection
    CFloatImage grayImage=ConvertToGray(image);

	int w = image.Shape().width;
    int h = image.Shape().height;

    for (vector<Feature>::iterator i = features.begin(); i != features.end(); i++) 
	{
        Feature &f = *i;

        int x = f.x;
        int y = f.y;

        f.data.resize(5 * 5);

        //TO DO---------------------------------------------------------------------
        // The descriptor is a 5x5 window of intensities sampled centered on the feature point.
		for (int i=0;i<5;i++)
		{
			for (int j=0;j<5;j++)
			{
				if (((y+i-2)>=0)&&((y+i-2)<h)&&((x+j-2)>=0)&&((x+j-2)<w))
				{
					// Keep in the same range for all descriptors
					f.data[j+5*i]=image.Pixel(x+j-2,y+i-2,1);
				}
				else
				{
					f.data[j+5*i]=0;
				}
			}
		}
    }
	//printf("ComputeSimpleDescriptors Done!\n"); 
}

// TODO: Implement parts of this function
// Compute MOPs descriptors.
void ComputeMOPSDescriptors(CFloatImage &image, FeatureSet &features)
{
    // This image represents the window around the feature you need to compute to store as the feature descriptor
    const int windowSize = 8;
    CFloatImage destImage(windowSize, windowSize, 1);

    for (vector<Feature>::iterator i = features.begin(); i != features.end(); i++) {
        Feature &f = *i;

        //TODO: Compute the inverse transform as described by the feature location/orientation.
        //You'll need to compute the transform from each pixel in the 8x8 image 
        //to sample from the appropriate pixels in the 40x40 rotated window surrounding the feature
        CTransform3x3 xform;

		// Values for computation
		double angle = f.angleRadians;
		int x = f.x;
		int y = f.y;

		// Take 41x41 square window around detected feature
		CFloatImage window(41, 41, 1);

		// Rotate to horizontal
		for (int i = 0; i < 41; i++)
			for (int j = 0; j < 41; j++)
			{
				// Relative x and y for points in the horizontal window
				int xx = x - 20 + i;
				int yy = y - 20 + j;
				// Corresponding x and y coordinates in the detected image
				// .5 is used for rounding
				int xPos = floor(cos(angle) * (xx - x) - sin(angle) * (yy - y) + .5) + x;
				int yPos = floor(sin(angle) * (xx - x) + cos(angle) * (yy - y) + .5) + y;
				// Check whether out of boundary
				if (!image.Shape().InBounds(xPos, yPos))
				{
					window.Pixel(i, j, 0) = 0;
					continue;
				}
				window.Pixel(i, j, 0) = image.Pixel(xPos, yPos, 1);
			}

		// Scale to 1/5 size(Prefiltering)
		// imageCopy is used to avoid prefiltering a prefiltered image
		CFloatImage imageCopy = image;
		CFloatImage gaussianKernel(5, 5, 1);
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				gaussianKernel.Pixel(i, j, 0) = gaussian5x5[5 * j + i];
		Convolve(window, image, gaussianKernel);
		// Sample
		xform[0][0] = 41 / 8;
		xform[1][1] = 41 / 8;
        //Call the Warp Global function to do the mapping
        WarpGlobal(image, destImage, xform, eWarpInterpLinear);

        f.data.resize(windowSize * windowSize);

        //TODO: fill in the feature descriptor data for a MOPS descriptor
		// Normalize intensity
		// 1) Compute mean
		double sum = 0.0;
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
				sum += destImage.Pixel(i, j, 0);
		double mean = sum / 64.0;
		// 2) Compute standard deviation
		double ssum = 0.0;
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
				ssum += pow(destImage.Pixel(i, j, 0) - mean, 2);
		double sdev = sqrt(ssum / 64);
		if (sdev == 0)
		{
			image = imageCopy;
			continue;
		}
		// 3) Fill in descriptor data
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
			// push_back will keep the initial values
			f.data[8 * i + j] = (destImage.Pixel(j, i, 0) - mean) / (sdev + 1e-10);
		image = imageCopy;
    }
	//printf("ComputeMOPSDescriptors Done!\n");
}

// Compute Custom descriptors (extra credit)
void ComputeCustomDescriptors(CFloatImage &image, FeatureSet &features)
{
	int w = image.Shape().width;
    int h = image.Shape().height;
	/* Scale Invariant To be Implemented
    CFloatImage oriFiltered(w, h, 3);
	CFloatImage halfFiltered(w/2, h/2, 1);
	CFloatImage scale_half(w/2, h/2, 1);
	CFloatImage scale_onefourth(w/4, h/4, 1);

	CFloatImage LPFilter(5, 5, 1);
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
			LPFilter.Pixel(i, j, 0) = gaussian5x5[5 * j + i];
	// Downsample
	Convolve(image, oriFiltered, LPFilter);
	for (int i = 0; i < scale_half.Shape().width; i++)
		for (int j = 0; j < scale_half.Shape().height; j++)
			scale_half.Pixel(i, j, 0) = oriFiltered.Pixel(i * 2, j * 2, 1);
	// Downsample
	Convolve(scale_half, halfFiltered, LPFilter);
	for (int i = 0; i < scale_onefourth.Shape().width; i++)
		for (int j = 0; j < scale_onefourth.Shape().height; j++)
			scale_onefourth.Pixel(i, j, 0) = halfFiltered.Pixel(i * 2, j * 2, 0);
	*/
    for (vector<Feature>::iterator i = features.begin(); i != features.end(); i++) 
	{
        Feature &f = *i;

        int x = f.x;
        int y = f.y;
		double angle = f.angleRadians;
		double step_angle = 20.0 / 180.0 * PI;
		
		f.data.resize(54);
		// Parameter r0 to decide radius of circle
		double r0 = 6.0;
		// 3 Levels with radius r0, 2r0 and 3r0
		for (int k = 1; k <= 3; k++)
		{
			for (int step = 0; step < 18; step++)
			{
				// Always start at the dominant orientation
				int xPos = x + floor(r0 * (double)k * cos(angle - step * step_angle) + .5);
				int yPos = y + floor(r0 * (double)k * sin(angle - step * step_angle) + .5);
				if (!image.Shape().InBounds(xPos, yPos))
				{
					f.data[step + 18 * (k - 1)] = 0;
					continue;
				}
				f.data[step + 18 * (k - 1)] = image.Pixel(xPos, yPos, 1);
				// Used for debug
				//if (i == features.begin())
				//{
					//cout << step_angle << endl;
					//cout << cos(angle) << " " << cos(angle-step_angle) << " " << cos(angle-2*step_angle) << endl;
					//cout << "x: " << x << ", y: " << y << ", cos: " << cos(angle) << ", sin: " << sin(angle) << endl;
					//cout << "xPos: " << xPos << ", yPos: " << yPos << ", data: " << image.Pixel(xPos, yPos, 1) << endl;
				//}
			}
		}
		// Normalize intensity
		// 1) Compute mean
		double sum = 0.0;
		for (int i = 0; i < 54; i++)
			sum += f.data[i];
		double mean = sum / 64.0;
		// 2) Compute standard deviation
		double ssum = 0.0;
		for (int i = 0; i < 54; i++)
			ssum += pow(f.data[i] - mean, 2);
		double sdev = sqrt(ssum / 64);
		// 3) Fill in descriptor data
		for (int i = 0; i < 54; i++)
			f.data[i] = (f.data[i] - mean) / (sdev + 1e-10);
	}
	printf("ComputeCustomDescriptors Done!\n");
}

// Perform simple feature matching.  This just uses the SSD
// distance between two feature vectors, and matches a feature in the
// first image with the closest feature in the second image.  It can
// match multiple features in the first image to the same feature in
// the second image.
void ssdMatchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches) {
    int m = f1.size();
    int n = f2.size();

    matches.resize(m);

    double d;
    double dBest;
    int idBest;

    for (int i=0; i<m; i++) {
        dBest = 1e100;
        idBest = 0;

        for (int j=0; j<n; j++) {
            d = distanceSSD(f1[i].data, f2[j].data);

            if (d < dBest) {
                dBest = d;
                idBest = f2[j].id;
            }
        }

        matches[i].id1 = f1[i].id;
        matches[i].id2 = idBest;
        matches[i].distance = dBest;
    }
}

//TODO: Write this function to perform ratio feature matching.  
// This just uses the ratio of the SSD distance of the two best matches
// and matches a feature in the first image with the closest feature in the second image.
// It can match multiple features in the first image to the same feature in
// the second image.  (See class notes for more information)
// You don't need to threshold matches in this function -- just specify the match distance
// in each FeatureMatch object, as well as the ids of the two matched features (see
// ssdMatchFeatures for reference).
void ratioMatchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches) 
{
	int m = f1.size();
	int n = f2.size();

	// Initialize matches space
	matches.resize(m);

	double d;
	double dBest1, dBest2;
	int idBest1, idBest2;

	for (int i = 0; i < m; i++)
	{
		// Initialize
		dBest1 = 1e100;
		idBest1 = 0;
		dBest2 = 1e100 + 1;
		idBest2 = 0;
		
		// Find closest in the second image using ratio of SSD distance
		for (int j = 0; j < n; j++)
		{
			d = distanceSSD(f1[i].data, f2[j].data);
			if (d < dBest1)
			{
				dBest2 = dBest1;
				dBest1 = d;
				idBest2 = idBest1;
				idBest1 = f2[j].id;
			}
			else if (d >= dBest1 && d < dBest2)
			{
				dBest2 = d;
				idBest2 = f2[j].id;
			}
		}

		matches[i].id1 = f1[i].id;
		matches[i].id2 = idBest1;
		matches[i].distance = dBest1 / dBest2;
	}
}

// Though the third best match may not compare to the first and the second
// in general, but it still contains useful information. By utilizing the
// ratio secondBest / thirdBest with a small weighting value and firstBest
// / secondBest with a relative big weighting value, we can take full
// advantage of the SSD information generated and distinguish between
// multiple ambiguous matches (proposed by Yingchuan)
// Tested and proved better performance than the original ratio match
void improvedratioMatchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches)
{
	int m = f1.size();
	int n = f2.size();
	double weight = 0.8;

	matches.resize(m);

	double d;
	double dBest1, dBest2, dBest3;
	int idBest1, idBest2, idBest3;
	for (int i = 0; i < m; i++)
	{
		dBest1 = 1e100;
		dBest2 = 1e100 + 1;
		dBest3 = 1e100 + 2;
		idBest1 = 0;
		idBest2 = 0;
		idBest3 = 0;
		for (int j = 0; j < n; j++)
		{
			d = distanceSSD(f1[i].data, f2[j].data);
			//cout << "d: " << d << endl;
			if (d < dBest1)
			{
				dBest3 = dBest2;
				dBest2 = dBest1;
				dBest1 = d;
				//cout << "dBest: " << dBest << endl;
				idBest3 = idBest2;
				idBest2 = idBest1;
				idBest1 = f2[j].id;
			}
			else if (d >= dBest1 && d < dBest2)
			{
				dBest3 = dBest2;
				dBest2 = d;
				idBest3 = idBest2;
				idBest2 = f2[j].id;
			}
			else if (d >= dBest2 && d < dBest3)
			{
				dBest3 = d;
				idBest3 = f2[j].id;
			}
		}
		matches[i].id1 = f1[i].id;
		matches[i].id2 = idBest1;
		matches[i].distance = weight * dBest1 / dBest2 + ( 1.0 - weight) * dBest2 / dBest3;
		//cout << "dBest: " << dBest << endl;
	}
}

// Convert Fl_Image to CFloatImage.
bool convertImage(const Fl_Image *image, CFloatImage &convertedImage) {
    if (image == NULL) {
        return false;
    }

    // Let's not handle indexed color images.
    if (image->count() != 1) {
        return false;
    }

    int w = image->w();
    int h = image->h();
    int d = image->d();

    // Get the image data.
    const char *const *data = image->data();

    int index = 0;

    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            if (d < 3) {
                // If there are fewer than 3 channels, just use the
                // first one for all colors.
                convertedImage.Pixel(x,y,0) = ((uchar) data[0][index]) / 255.0f;
                convertedImage.Pixel(x,y,1) = ((uchar) data[0][index]) / 255.0f;
                convertedImage.Pixel(x,y,2) = ((uchar) data[0][index]) / 255.0f;
            }
            else {
                // Otherwise, use the first 3.
                convertedImage.Pixel(x,y,0) = ((uchar) data[0][index]) / 255.0f;
                convertedImage.Pixel(x,y,1) = ((uchar) data[0][index+1]) / 255.0f;
                convertedImage.Pixel(x,y,2) = ((uchar) data[0][index+2]) / 255.0f;
            }

            index += d;
        }
    }

    return true;
}

// Convert CFloatImage to CByteImage.
void convertToByteImage(CFloatImage &floatImage, CByteImage &byteImage) {
    CShape sh = floatImage.Shape();

    assert(floatImage.Shape().nBands == byteImage.Shape().nBands);
    for (int y=0; y<sh.height; y++) {
        for (int x=0; x<sh.width; x++) {
            for (int c=0; c<sh.nBands; c++) {
                float value = floor(255*floatImage.Pixel(x,y,c) + 0.5f);

                if (value < byteImage.MinVal()) {
                    value = byteImage.MinVal();
                }
                else if (value > byteImage.MaxVal()) {
                    value = byteImage.MaxVal();
                }

                // We have to flip the image and reverse the color
                // channels to get it to come out right.  How silly!
                byteImage.Pixel(x,sh.height-y-1,sh.nBands-c-1) = (uchar) value;
            }
        }
    }
}

// Compute SSD distance between two vectors.
double distanceSSD(const vector<double> &v1, const vector<double> &v2) {
    int m = v1.size();
    int n = v2.size();

    if (m != n) {
        // Here's a big number.
        return 1e100;
    }

    double dist = 0;

    for (int i=0; i<m; i++) {
        dist += pow(v1[i]-v2[i], 2);
    }


    return sqrt(dist);
}

// Transform point by homography.
void applyHomography(double x, double y, double &xNew, double &yNew, double h[9]) {
    double d = h[6]*x + h[7]*y + h[8];

    xNew = (h[0]*x + h[1]*y + h[2]) / d;
    yNew = (h[3]*x + h[4]*y + h[5]) / d;
}

// Evaluate a match using a ground truth homography.  This computes the
// average SSD distance between the matched feature points and
// the actual transformed positions.
double evaluateMatch(const FeatureSet &f1, const FeatureSet &f2, const vector<FeatureMatch> &matches, double h[9]) {
    double d = 0;
    int n = 0;

    double xNew;
    double yNew;

    unsigned int num_matches = matches.size();
    for (unsigned int i=0; i<num_matches; i++) {
        int id1 = matches[i].id1;
        int id2 = matches[i].id2;
        applyHomography(f1[id1].x, f1[id1].y, xNew, yNew, h);
        d += sqrt(pow(xNew-f2[id2].x,2)+pow(yNew-f2[id2].y,2));
        n++;
    }	

    return d / n;
}

void addRocData(const FeatureSet &f1, const FeatureSet &f2, const vector<FeatureMatch> &matches, double h[9],
    vector<bool> &isMatch, double threshold, double &maxD) 
{
    double d = 0;

    double xNew;
    double yNew;

    unsigned int num_matches = matches.size();
    for (unsigned int i=0; i<num_matches; i++) {
        int id1 = matches[i].id1;
        int id2 = matches[i].id2;
        applyHomography(f1[id1].x, f1[id1].y, xNew, yNew, h);

        // Ignore unmatched points.  There might be a better way to
        // handle this.
        d = sqrt(pow(xNew-f2[id2].x,2)+pow(yNew-f2[id2].y,2));
        if (d<=threshold) {
            isMatch.push_back(1);
        } else {
            isMatch.push_back(0);
        }

        if (matches[i].distance > maxD)
            maxD = matches[i].distance;
    }	
}

vector<ROCPoint> computeRocCurve(vector<FeatureMatch> &matches,vector<bool> &isMatch,vector<double> &thresholds)
{
    vector<ROCPoint> dataPoints;

    for (int i=0; i < (int)thresholds.size();i++)
    {
        //printf("Checking threshold: %lf.\r\n",thresholds[i]);
        int tp=0;
        int actualCorrect=0;
        int fp=0;
        int actualError=0;
        int total=0;

        int num_matches = (int) matches.size();
        for (int j=0;j < num_matches;j++) {
            if (isMatch[j]) {
                actualCorrect++;
                if (matches[j].distance < thresholds[i]) {
                    tp++;
                }
            } else {
                actualError++;
                if (matches[j].distance < thresholds[i]) {
                    fp++;
                }
            }

            total++;
        }

        ROCPoint newPoint;
        //printf("newPoints: %lf,%lf",newPoint.trueRate,newPoint.falseRate);
        newPoint.trueRate=(double(tp)/actualCorrect);
        newPoint.falseRate=(double(fp)/actualError);
        //printf("newPoints: %lf,%lf",newPoint.trueRate,newPoint.falseRate);

        dataPoints.push_back(newPoint);
    }

    return dataPoints;
}



// Compute AUC given a ROC curve
double computeAUC(vector<ROCPoint> &results)
{
    double auc=0;
    double xdiff,ydiff;
    for (int i = 1; i < (int) results.size(); i++)
    {
        //fprintf(stream,"%lf\t%lf\t%lf\n",thresholdList[i],results[i].falseRate,results[i].trueRate);
        xdiff=(results[i].falseRate-results[i-1].falseRate);
        ydiff=(results[i].trueRate-results[i-1].trueRate);
        auc=auc+xdiff*results[i-1].trueRate+xdiff*ydiff/2;

    }
    return auc;
}

