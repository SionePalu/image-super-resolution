import numpy as np

from ISR.utils.image_processing import (
    process_array,
    process_output,
    split_image_into_overlapping_patches,
    stich_together,
)

from multiprocessing import Process, Manager
import multiprocessing as mp
from time import sleep

def predictSinglePatch(listOfLists: list, isrModel: ImageModel):
    patchIdx = listOfLists[0]
    print('patchIdx#', str(patchIdx))
    patchArray = listOfLists[1]
    predictedPatchArray = isrModel.predict(patchArray)
    returnList = [patchIdx, predictedPatchArray]
    return returnList

class ImageModel:
    """ISR models parent class.

    Contains functions that are common across the super-scaling models.
    """
    
    def predict(self, input_image_array, imgInfo='', by_patch_of_size=None, batch_size=10, padding_size=2):
        """
        Processes the image array into a suitable format
        and transforms the network output in a suitable image format.

        Args:
            input_image_array: input image array.
            by_patch_of_size: for large image inference. Splits the image into
                patches of the given size.
            padding_size: for large image inference. Padding between the patches.
                Increase the value if there is seamlines.
            batch_size: for large image inferce. Number of patches processed at a time.
                Keep low and increase by_patch_of_size instead.
        Returns:
            sr_img: image output.
        """
        
        if by_patch_of_size:
            lr_img = process_array(input_image_array, expand=False)
            patches, p_shape = split_image_into_overlapping_patches(
                lr_img, patch_size=by_patch_of_size, padding_size=padding_size
            )
            numPatches = len(patches)
            print('patches shape: ', list(patches.shape))
            print('p_shape shape: ', list(p_shape))
            
            if len(imgInfo) > 0 and imgInfo == 'domap':
               print('parallel processing')
               manager = Manager()
               listOfLists = manager.list()
               for i in range(0, len(patches), batch_size):
                   patchIdx = i
                   patchArray = patches[i: i + batch_size]
                   listOfLists.append([patchIdx, patchArray])
               
               ncpu = mp.cpu_count()
               print("Number of processors: ", ncpu)
               print(' ')
           
               pool = mp.Pool(ncpu)
#     predictSinglePatch(listOfLists: list, isrModel: ImageModel)
               with Pool(processes=ncpu) as pool:
                  poolObjList = pool.starmap(predictSinglePatch, zip( listOfLists, repeat(self) ))
                  
                  for objL in poolObjList:
                    if len(objL) > 0:
                      self.results.append(objL)
            
                  pool.close()
                 
            else:
               # return patches
               for i in range(0, len(patches), batch_size):
                   patstr = imgInfo + "Patch-" + str(i+1) + " of " + str(numPatches)
                   print(patstr)
                   batch = self.model.predict(patches[i: i + batch_size])
                   if i == 0:
                       collect = batch
                   else:
                       collect = np.append(collect, batch, axis=0)
                       print('collect shape: ', list(collect.shape))
            
            scale = self.scale
            padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
            scaled_image_shape = tuple(np.multiply(input_image_array.shape[0:2], scale)) + (3,)
            sr_img = stich_together(
                collect,
                padded_image_shape=padded_size_scaled,
                target_shape=scaled_image_shape,
                padding_size=padding_size * scale,
            )
        
        else:
            lr_img = process_array(input_image_array)
            sr_img = self.model.predict(lr_img)[0]
        
        sr_img = process_output(sr_img)
        return sr_img
