import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# import skimage
from skimage.morphology import binary_closing, remove_small_holes, remove_small_objects, ball
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import disk, circle_perimeter
from skimage.feature import canny
from skimage.exposure import equalize_hist
from skimage.measure import label


FIRST_THRESHOLD = 150
LAST_THRESHOLD = 501
THRESHOLD_STEP = 14
UPPER_THRESHOLD = 1300
AURTA_X = [200, 260]
AURTA_Y = [210, 300]
CASE_1 = 1
CASE_2 = 2
CASE_3 = 3
CASES = [CASE_1, CASE_2, CASE_3]

def read_nifty(path):
    """
    gets a nifti file and returns a numpy array containing the image
    :param path: path of nifty file
    :return: numpy nd array of image
    """
    seg_nifti_file = nib.load(f"{path}")
    img = seg_nifti_file.get_fdata()
    return img



class BonesSegmentation:
    """
    skeleton segmentation class
    """
    def __init__(self, nifty_file_path):
        self.nifty_file_path = nifty_file_path


    def SkeletonTHFinder(self):
        """
        takes the CT in the nifty_file and segments the skeleton using threshold and morphological operations
        :return: saves the skeleton segmentation
        """
        best_threshold = int(self.get_best_threshold())
        print(f"best threshold is {best_threshold}")
        selected_file = nib.load(f"{self.nifty_file_path}_seg_{best_threshold}_{UPPER_THRESHOLD}.nii.gz")
        image_3D = selected_file.get_fdata()
        image_3D = image_3D.astype(int)
        mult_val = 1000000

        # post-processing
        best_cc = np.inf
        stop_flag = 1
        image = image_3D
        while best_cc > 1:
            area = stop_flag * mult_val
            image = remove_small_holes(image, area, connectivity=3)
            image = remove_small_objects(image, (area * 4) / mult_val, connectivity=3)
            image = binary_closing(image)

            prev_cc = best_cc
            best_cc = label(image, return_num=True)[1]
            stop_flag += 1
            print(f"best_cc = {best_cc}, area = {area}")
            if (best_cc == prev_cc and stop_flag > 4) or stop_flag > 10:
                break

        best_img = image
        # save
        new_file = nib.Nifti1Image(best_img.astype(int), selected_file.affine)
        new_file_path = f"{self.nifty_file_path}_SkeletonSegmentation.nii.gz"
        nib.save(new_file, new_file_path)


    def get_best_threshold(self):
        """
        Searching through the threshold space and returning the threshold that gives the lowest number of
        connected components in the 3D CT cimage
        :return: int representing the best minimum threshold
        """
        steps = (LAST_THRESHOLD - FIRST_THRESHOLD) // THRESHOLD_STEP + 1
        ccs_per_threshold = np.zeros((steps, 2))
        for step_num, lower_threshold in enumerate(np.arange(FIRST_THRESHOLD, LAST_THRESHOLD, THRESHOLD_STEP)):
            # do thresholding and save the file
            Imin = lower_threshold
            Imax = UPPER_THRESHOLD
            self.SegmentationByTH(Imin, Imax)
            # extract file again and data, check number of connectivity components
            seg_nifti_file = nib.load(f"{self.nifty_file_path}_seg_{Imin}_{Imax}.nii.gz")
            seg_3d_img = seg_nifti_file.get_fdata()
            _, ccs_num = label(seg_3d_img, background=None, return_num=True, connectivity=3)
            ccs_per_threshold[step_num, 0] = lower_threshold
            ccs_per_threshold[step_num, 1] = ccs_num
            print(lower_threshold, ccs_num)
        xdata = ccs_per_threshold[:, 0]
        ydata = ccs_per_threshold[:, 1]
        ccs = ccs_per_threshold[:, 1]
        min_cc_inx = np.argmin(ccs)
        best_threshold = ccs_per_threshold[min_cc_inx, 0]
        self.plot_ccns_per_threshold(best_threshold, xdata, ydata)
        return best_threshold

    def plot_ccns_per_threshold(self, best_threshold, xdata, ydata):
        """
        displays number of connectivity components per Imin threshold
        :param best_threshold
        :param xdata: lower_thresholds
        :param ydata: number of connectivity components
        :return: plot
        """
        plt.plot(xdata, ydata)
        plt.title(f"number of connectivity components per Imin threshold \nbest threshold = {best_threshold}")
        plt.xlabel('Imin threshold')
        plt.ylabel('number of connectivity components')
        plt.grid()
        plt.show()

    def SegmentationByTH(self, Imin, Imax):
        """
        loads the CT image and thresholds it according to Imin and Imax values
        :param nifty_file: grayscale NIFTI file path
        :param Imin: minimal threshold
        :param Imax: maximal threshold
        :return:
        """
        # get 3d image data
        try:
            nifti_file = nib.load( self.nifty_file_path)
            img_data = nifti_file.get_fdata()
            # TODO add errors
        except:
            return 0
        # threshold
        img_data = np.where((img_data >= Imin) & (img_data <= Imax), 1, 0)
        # save new file
        new_nifti = nib.Nifti1Image(img_data, nifti_file.affine)
        new_file_path = f"{self.nifty_file_path}_seg_{int(Imin)}_{int(Imax)}.nii.gz"
        nib.save(new_nifti, new_file_path)
        return 1


class AortaSegmentation:

    """
    Class for Aorta segmentation
    """
    def find_case(self):
        case = None
        for i in range(1, 4):
            if f"Case{i}" in self.path_CT_nifti_file:
                case = CASES[i - 1]
        return case


    def get_L1_boundaries(self):
        """
        find x,y,z boundaries of L1
        :return: min_x, max_x, min_y, max_y, min_z, max_z
        """
        z_idx = np.where(self.L1_img != 0)[2]
        y_idx = np.where(self.L1_img != 0)[1]
        x_idx = np.where(self.L1_img != 0)[0]
        min_x = np.min(x_idx)
        max_x = np.max(x_idx)
        min_y = np.min(y_idx)
        max_y = np.max(y_idx)
        min_z = np.min(z_idx)
        max_z = np.max(z_idx)
        return min_x, max_x, min_y, max_y, min_z, max_z


    def __init__(self, path_CT_nifti_file, path_L1_seg_nifti_file, path_GT):
        self.path_CT_nifti_file = path_CT_nifti_file
        self.path_L1_seg_nifti_file = path_L1_seg_nifti_file
        self.path_GT = path_GT
        self.affine = nib.load(f"{path_CT_nifti_file}").affine
        self.L1_img = read_nifty(self.path_L1_seg_nifti_file)
        self.CT_img = read_nifty(self.path_CT_nifti_file)
        self.case = self.find_case()
        self.L1_min_x, self.L1_max_x, self.L1_min_y, self.L1_max_y, self.L1_min_z, self.L1_max_z = self.get_L1_boundaries()
        self.cut_CT_img = self.CT_img[:, :, self.L1_min_z: self.L1_max_z + 1]
        self.num_slices =  self.cut_CT_img.shape[-1]
        self.search_x = [self.L1_min_x, self.L1_max_x]
        self.search_y = [self.L1_min_y - 70, self.L1_min_y + 20]
        self.slice_3d_img = self.cut_CT_img[self.search_x[0]:self.search_x[1], self.search_y[0]:self.search_y[1], :]
        self.default_radii = np.arange(7, 20, 1)
        self.case_3_radii = np.arange(10, 13, 1)
        self.num_circles_to_search = 20
        self.ignore_circle_error_rate = 10
        self.binary_closing_fp = 10


    def AortaSegmentation(self):
        """
        segment Aorta image and save segmentation
        :return:
        """
        segmented_Aorta_L1 = self.find_Aorta()
        self.save_Aorta_segmentation(segmented_Aorta_L1)
        return segmented_Aorta_L1

    def save_Aorta_segmentation(self, segmented_Aorta_L1):
        """
        saves Aorta segmentation
        :param segmented_Aorta_L1: 3d numpy array of segmentation
        :return: None
        """
        new_file = nib.Nifti1Image(segmented_Aorta_L1.astype(int), self.affine)
        file_name = self.path_CT_nifti_file.split('.')[0]
        new_file_path = f"{file_name}_Aorta_Segmentation.nii.gz"
        nib.save(new_file, new_file_path)

    def find_Aorta(self):
        """
        finds all Aorta pixels in CT: first per slice and then for all 3D image
        :return: segmented Aorta
        """
        circles = self.find_possible_Aorta_circles()
        # find best circle in each slice based on previous slice
        final_circles = np.array(self.extract_best_Aorta_circles(circles))
        # self.display_final_circles(final_circles)
        segmented_Aorta = self.construct_3d_Aurta_segmentation(final_circles)
        self.display_Aorta_on_CT(segmented_Aorta)
        return segmented_Aorta



    def find_possible_Aorta_circles(self):
        """
        for each L1 slice finds self.num_circles_to_search possible Aorta's circle locations
        using hough transform
        :return: circles: a list of circles information arrays per slice
        """
        circles = []
        slice_3d_img = self.cut_CT_img[self.search_x[0]:self.search_x[1], self.search_y[0]:self.search_y[1], :]
        for slice in range(self.num_slices):
            slice_2d_img = slice_3d_img[:, :, slice]
            slice_2d_img = equalize_hist(slice_2d_img)
            edges = canny(slice_2d_img, sigma=3).astype(int)
            if self.case is CASE_3 and slice == 0:
                radii = self.case_3_radii
            else:
                radii = self.default_radii
            hspaces = hough_circle(edges, radii)
            accums, cx, cy, rad = hough_circle_peaks(hspaces, radii, total_num_peaks=self.num_circles_to_search)
            circles.append(np.array([cx, cy, rad]))
            # self.display_all_circles_on_slice(cx, cy, rad, slice_2d_img)
        return circles


    def extract_best_Aorta_circles(self, circles):
        """
        extract the most fitting Aorta circle for each slice
        :param circles: the output of find_possible_Aorta_circles
        :return: final_circles: an array of the selected 1 circle per slice for Aorta location
        """
        final_circles = []
        prev_circle = None
        eps = self.ignore_circle_error_rate
        for slice in range(self.num_slices):
            slice_circles = circles[slice]
            num_circles = slice_circles.shape[1]
            if slice == 0:
                first_cirlce = slice_circles[:, 0]
                final_circles.append(first_cirlce)
                prev_circle = first_cirlce
            else:
                best_dist = np.inf
                best_circle = None
                for circle_num in range(num_circles):
                    cur_circle = slice_circles[:, circle_num]
                    dist_to_prev = np.linalg.norm(cur_circle - prev_circle)
                    if dist_to_prev < best_dist:
                        best_circle = cur_circle
                        best_dist = dist_to_prev
                if best_dist > eps:  # if there is no good circle then just take previous circle
                    best_circle = prev_circle
                final_circles.append(best_circle)
                prev_circle = best_circle
        return final_circles


    def construct_3d_Aurta_segmentation(self, final_circles):
        """
        construct 3D Aorta segmentation from Aorta circles information from each slice
        :param final_circles: Aorta circles information from each slice
        :return: 3D Aorta segmentation
        """
        segmented_Aorta_cut = np.zeros(self.slice_3d_img.shape)
        for slice in range(self.num_slices):
            slice_location = slice
            # disk coordinates
            center = final_circles[slice, [0, 1]]
            radius = final_circles[slice, 2]
            ys, xs = disk(center, radius)
            segmented_Aorta_cut[xs, ys, slice_location] = 1
        segmented_Aorta_cut = binary_closing(segmented_Aorta_cut, footprint=ball(radius=self.binary_closing_fp))
        segmented_Aorta = np.zeros(self.CT_img.shape)
        segmented_Aorta[self.search_x[0]:self.search_x[1], self.search_y[0]:self.search_y[1], self.L1_min_z: self.L1_max_z + 1] = segmented_Aorta_cut
        return segmented_Aorta

    @staticmethod
    def evaluateSegmentation(GT_seg, est_seg):
        """
        evaluate performance of segmentation compared to the ground truth segmentation
        :param GT_seg: ground truth segmentation
        :param est_seg: computed segmentation
        :return: DICE ((2 * intersection_size) / (size_A + size_B)) and VOD (1 - intersection_size/union_size) scores
        """
        # cut according to z boundaries
        z_idx = np.where(est_seg != 0)[2]
        min_z = np.min(z_idx)
        max_z = np.max(z_idx)

        # cut images according to relevant area
        GT_seg_cut, est_seg_cut = GT_seg[:, :, min_z:max_z + 1], est_seg[:, :, min_z:max_z + 1]
        intersection_image = np.logical_and(GT_seg_cut, est_seg_cut).astype(int)
        intersection_size = np.count_nonzero(intersection_image)

        size_A = np.count_nonzero(GT_seg_cut)
        size_B = np.count_nonzero(est_seg_cut)

        dice = (2 * intersection_size) / (size_A + size_B)

        union_image = np.logical_or(GT_seg_cut, est_seg_cut).astype(int)
        union_size = np.count_nonzero(union_image)

        vod = 1 - intersection_size/union_size
        return vod, dice



    def display_all_circles_on_slice(self, c_x, c_y, rad_, slice_img):
        """
        for each slice displays all the circles found by hough transform
        """
        from skimage.draw import circle_perimeter
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        for center_y, center_x, radius in zip(c_x, c_y, rad_):
            # print(radius)
            circy, circx = circle_perimeter(center_y, center_x, radius,
                                            shape=slice_img.shape)
            circx_new = circx
            circy_new = circy
            circx_new[circx >= slice_img.shape[0]] = 0
            circy_new[circx >= slice_img.shape[0]] = 0
            circx_new[circy >= slice_img.shape[1]] = 0
            circy_new[circy >= slice_img.shape[1]] = 0
            slice_img[circx_new, circy_new] = (1)
        ax.imshow(slice_img, cmap=plt.cm.gray)
        plt.show()

    def display_Aorta_on_CT(self, segmented_Aorta):
        """
        displays every slice of relevant CT with the segmented Aorta on top of it
        :param segmented_Aorta:
        :return:
        """
        for slice in range(self.num_slices):
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
            slice_2d_img = segmented_Aorta[:, :, slice + self.L1_min_z]
            ax.imshow(self.CT_img[:, :, slice + self.L1_min_z] - slice_2d_img * 1000, cmap=plt.cm.gray)
            plt.show()

    def display_final_circles(self, final_circles):
        """
        displays the one chosen circle per slice represents the Aorta location
        :param final_circles: array of circles information
        """
        from skimage.draw import circle_perimeter
        for slice in range(self.num_slices):
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
            slice_2d_img = self.slice_3d_img[:, :, slice]
            best_circle = final_circles[slice]
            circy, circx = circle_perimeter(best_circle[0], best_circle[1], best_circle[2],
                                            shape=slice_2d_img.shape)
            slice_2d_img[circx, circy] = 0
            ax.imshow(slice_2d_img, cmap=plt.cm.gray)
            plt.show()




def run_Aorta_segmentation():

    L1_case1 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\case 1\Case1_L1.nii.gz"
    CT_case1 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\case 1\Case1_CT.nii.gz"

    L1_case2 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case2_L1.nii.gz"
    CT_case2 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case2_CT.nii.gz"

    L1_case3 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case3_L1.nii.gz"
    CT_case3 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case3_CT.nii.gz"

    L1_case4 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case4_L1.nii.gz"
    CT_case4 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case4_CT.nii.gz"


    A = AortaSegmentation(CT_case1, L1_case1,  None)
    est_segmentation =  A.AortaSegmentation()
    print("finish case 1")
    A = AortaSegmentation(CT_case2, L1_case2,  None)
    est_segmentation =  A.AortaSegmentation()
    print("finish case 2")
    A = AortaSegmentation(CT_case3, L1_case3,  None)
    est_segmentation =  A.AortaSegmentation()
    print("finish case 3")
    A = AortaSegmentation(CT_case4, L1_case4,  None)
    est_segmentation =  A.AortaSegmentation()
    print("finish case 4")


    path_est_segmentation_1 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\case 1\Case1_CT_Aorta_Segmentation.nii.gz"
    GT_path_1 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\case 1\Case1_Aorta.nii.gz"

    path_est_segmentation_2 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case2_CT_Aorta_Segmentation.nii.gz"
    GT_path_2 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case2_Aorta.nii.gz"

    path_est_segmentation_3 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case3_CT_Aorta_Segmentation.nii.gz"
    GT_path_3 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case3_Aorta.nii.gz"

    path_est_segmentation_4 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case4_CT_Aorta_Segmentation.nii.gz"
    GT_path_4 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\Case4_Aorta.nii.gz"


    est_seg = read_nifty(path=path_est_segmentation_1)
    GT_seg = read_nifty(path=GT_path_1)
    vod, dice = AortaSegmentation.evaluateSegmentation(GT_seg, est_seg)
    print(f"case 1 dice = {dice}, vod = {vod}")

    est_seg = read_nifty(path=path_est_segmentation_2)
    GT_seg = read_nifty(path=GT_path_2)
    vod, dice = AortaSegmentation.evaluateSegmentation(GT_seg, est_seg)
    print(f"case 2 dice = {dice}, vod = {vod}")

    est_seg = read_nifty(path=path_est_segmentation_3)
    GT_seg = read_nifty(path=GT_path_3)
    vod, dice = AortaSegmentation.evaluateSegmentation(GT_seg, est_seg)
    print(f"case 3 dice = {dice}, vod = {vod}")

    est_seg = read_nifty(path=path_est_segmentation_4)
    GT_seg = read_nifty(path=GT_path_4)
    vod, dice = AortaSegmentation.evaluateSegmentation(GT_seg, est_seg)
    print(f"case 4 dice = {dice}, vod = {vod}")


def run_skeleton_segmettation():
    path_1 = rf"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\case 1\Case1_CT.nii.gz"
    path_2 = rf"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\case 2\Case2_CT.nii.gz"
    path_3 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\case 3\Case3_CT.nii.gz"
    path_4 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\case 4\Case4_CT.nii.gz"
    path_5 = r"C:\Users\amita\PycharmProjects\pythonProject\pythonProject\MIP_ex1\Mip Data Targil 2\case 5\Case5_CT.nii.gz"

    B = BonesSegmentation(path_1)
    B.SkeletonTHFinder()
    B = BonesSegmentation(path_2)
    B.SkeletonTHFinder()
    B = BonesSegmentation(path_3)
    B.SkeletonTHFinder()
    B = BonesSegmentation(path_4)
    B.SkeletonTHFinder()
    B = BonesSegmentation(path_5)
    B.SkeletonTHFinder()
    print("finish case 5 skeleton segmentation")


"""
case 1 dice = 0.9326823494599793, vod = 0.12614360964790683
case 2 dice = 0.8205641650705207, vod = 0.3042740090056839
case 3 dice = 0.8749969968527016, vod = 0.22222696792380303
case 4 dice = 0.909468870179815, vod = 0.1660312619138391
"""

if __name__ == '__main__':
    run_Aorta_segmentation()
    run_skeleton_segmettation()
