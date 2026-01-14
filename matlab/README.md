# Laserdanger
 Lidar processing and supporting codes
 Currently set up to process point cloud scans in .laz format from a Livox Avia
 L1_pipeline is the main code for producing 30minute average DEM-like vectorized surfaces from the point clouds. 
 ---> utilizes accumarray and delauney triangulation to clean and bin point clouds, then return a structured output containing 
    - minimum, maximum, mode, and mean  surface  (min,mode, etc of points calculated for each bin) 
 ---> the data goes to a struct organized by day
 
 Get1D_profiles takes X,Y,Z data from L1_pipeline and outputs a 1D tprofile along a center transect, or several transects in the longshore. 
 ---> use it in a for loop to create a matrix Z(x,t) of 1D profiles over time. 

 L2_pipeline_testing is the current version of codes to produce matrices Z(x,t) at 2Hz from the point cloud scans and define the runup edge. 
 ---> raw point clouds are inputted, and data along a centered transect is accumulated to Z,X,Y,I (t) arrays, then formatted as 1D transects using 
 Get_1D_profiles_swash_accum 
 get_runupStats_L2 produces a runup edge in spatial coordinates and vertical elevations. 


%%%%%%
 Codes were made by Ashton Domi for his PhD thesis 
 with the Coastal Processes Group,
 Scripps Institution of Oceanography
