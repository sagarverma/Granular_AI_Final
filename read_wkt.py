import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
from os import listdir


def _get_image_names(base_path, imageId):
    '''
    Get the names of the tiff files
    '''
    d = {'3': path.join(base_path,'three_band/{}.tif'.format(imageId)),             # (3, 3348, 3403)
         'A': path.join(base_path,'sixteen_band/{}_A.tif'.format(imageId)),         # (8, 134, 137)
         'M': path.join(base_path,'sixteen_band/{}_M.tif'.format(imageId)),         # (8, 837, 851)
         'P': path.join(base_path,'sixteen_band/{}_P.tif'.format(imageId)),         # (3348, 3403)
         }
    return d


def _convert_coordinates_to_raster(coords, img_size, xymax):
    Xmax,Ymax = xymax
    H,W = img_size
    W1 = 1.0*W*W/(W+1)
    H1 = 1.0*H*H/(H+1)
    xf = W1/Xmax
    yf = H1/Ymax
    coords[:,1] *= yf
    coords[:,0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)
    return (xmax,ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list,interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value = 1):
    img_mask = np.zeros(raster_img_size,np.uint8)
    if contours is None:
        return img_mask
    perim_list,interior_list = contours
    cv2.fillPoly(img_mask,perim_list,class_value)
    cv2.fillPoly(img_mask,interior_list,0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda,
                                     wkt_list_pandas):
    xymax = _get_xmax_ymin(grid_sizes_panda,imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas,imageId,class_type)
    contours = _get_and_convert_contours(polygon_list,raster_size,xymax)
    mask = _plot_mask_from_contours(raster_size,contours,1)
    return mask


inDir = '../../datasets/DSTL'


# read the training data from train_wkt_v4.csv
df = pd.read_csv(inDir + '/train_wkt_v4.csv')

# grid size will also be needed later..
gs = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

images = listdir(inDir + '/three_band_pngs')

label_colours = cv2.imread('camvid12.png').astype(np.uint8)

for image in images:
    img = cv2.imread(inDir + '/three_band_pngs/' + image)
    nimg = np.zeros((img.shape[0],img.shape[1],3))
    mask0 = generate_mask_for_image_and_class((img.shape[0],img.shape[1]),image[:-4],0,gs,df)
    mask1 = generate_mask_for_image_and_class((img.shape[0],img.shape[1]),image[:-4],1,gs,df)
    mask2 = generate_mask_for_image_and_class((img.shape[0],img.shape[1]),image[:-4],2,gs,df)
    mask3 = generate_mask_for_image_and_class((img.shape[0],img.shape[1]),image[:-4],3,gs,df)
    mask4 = generate_mask_for_image_and_class((img.shape[0],img.shape[1]),image[:-4],4,gs,df)
    mask5 = generate_mask_for_image_and_class((img.shape[0],img.shape[1]),image[:-4],5,gs,df)
    mask6 = generate_mask_for_image_and_class((img.shape[0],img.shape[1]),image[:-4],6,gs,df)
    mask7 = generate_mask_for_image_and_class((img.shape[0],img.shape[1]),image[:-4],7,gs,df)
    mask8 = generate_mask_for_image_and_class((img.shape[0],img.shape[1]),image[:-4],8,gs,df)
    mask9 = generate_mask_for_image_and_class((img.shape[0],img.shape[1]),image[:-4],9,gs,df)
    mask10 = generate_mask_for_image_and_class((img.shape[0],img.shape[1]),image[:-4],10,gs,df)
    
    mask0[mask0 == 1] = 1
    mask0[mask1 == 1] = 2
    mask0[mask2 == 1] = 3
    mask0[mask3 == 1] = 4
    mask0[mask4 == 1] = 5
    mask0[mask5 == 1] = 6
    mask0[mask6 == 1] = 7
    mask0[mask7 == 1] = 8
    mask0[mask8 == 1] = 9
    mask0[mask9 == 1] = 10
    mask0[mask10 == 1] = 11
    
    if np.count_nonzero(mask0) > 0:
        nimg[mask0 == 0] = label_colours[0][0]
        nimg[mask0 == 1] = label_colours[0][1]
        nimg[mask0 == 2] = label_colours[0][2]
        nimg[mask0 == 3] = label_colours[0][3]
        nimg[mask0 == 4] = label_colours[0][4]
        nimg[mask0 == 5] = label_colours[0][5]
        nimg[mask0 == 6] = label_colours[0][6]
        nimg[mask0 == 7] = label_colours[0][7]
        nimg[mask0 == 8] = label_colours[0][8]
        nimg[mask0 == 9] = label_colours[0][9]
        nimg[mask0 == 10] = label_colours[0][10]
        nimg[mask0 == 11] = label_colours[0][11]

        #cv2.imwrite('../../datasets/DSTL/annotated_imgs/' + image ,nimg)
        cv2.imwrite('../../datasets/DSTL/train_three_band/' + image[:-4] + '_mask.png', mask0)
        cv2.imwrite('../../datasets/DSTL/train_three_band/' + image, img)