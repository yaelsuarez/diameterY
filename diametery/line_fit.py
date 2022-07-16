from matplotlib import lines
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as plt
from typing import Tuple, List
import cv2

point = Tuple[float,float]
line = Tuple[point, point]
measurements = List[line]

class LineFit:
    def __init__(self, n: int, step_size:float) -> None:
        """Model that fits a line in a binary image and measures diameter of fibers

        :param n: number of measurements to do along the fitted line.
        :param step_size: step size of diameter measurement (in pixels). Can be fraction.
        """
        self.n = n
        self.step_size = step_size

    def get_coordinates(self, im, value_for_mask):
        #I = rgb2gray(I_orig) #we can delete this if we get binary images
        mask = im > value_for_mask
        fiber_coor = np.argwhere(mask)
        x = fiber_coor[:, 1]
        y = fiber_coor[:, 0]
        return x, y

    def func_line(self, x, a, b):
        return a * x + b

    def func_line_inv(self, y, a, b):
        return (y - b)/a

    def get_fited_line_x_y(self, im):
        value_for_mask = (int(np.max(im))+int(np.min(im)))/2 # Pixels to mask in get_coordinate
        x, y = self.get_coordinates(im, value_for_mask)
        popt, pcov = curve_fit(self.func_line, x, y)
        return x, y, popt, pcov

    def get_fited_line_y_x(self, im):
        value_for_mask = (int(np.max(im))+int(np.min(im)))/2 # Pixels to mask in get_coordinate
        x, y = self.get_coordinates(im, value_for_mask)
        popt, pcov = curve_fit(self.func_line, y, x)
        return x, y, popt, pcov

    def get_better_fit(self, x, y, popt, popt_inv, pcov, pcov_inv):
        diagonal = np.diagonal(pcov)
        diagonal_inv = np.diagonal(pcov_inv)
        if np.less(diagonal, diagonal_inv).all() == True:
            popt_fit = popt
            x_line = np.arange(0, max(x), 1)
            y_line = []
            for i in x_line:
                a = self.func_line(x_line[i], *popt)
                y_line.append(a)
            y_fit = y_line
            x_fit = x_line
            p1 = [x_fit[0],y_fit[0]]
            p2 = [x_fit[-1],y_fit[-1]]
        elif np.less(diagonal, diagonal_inv).all() == False:
            popt_fit = [1/popt_inv[0], (-popt_inv[1])/popt_inv[0]]
            y_line = np.arange(0, max(y), 1)
            x_line = []
            for i in y_line:
                a = self.func_line(y_line[i], *popt_inv)
                x_line.append(a)
            y_fit = y_line
            x_fit = x_line
            p1 = [x_fit[0],y_fit[0]]
            p2 = [x_fit[-1],y_fit[-1]]
        else:
            print("One of the pcov values is True and the rest are False")
        return popt_fit, x_fit, y_fit, p1, p2

    def get_point(self, t, p1, p2):
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        p = [(dx * t + p1[0]), (dy * t + p1[1])]
        return p, dx, dy

    def get_normal_vector(self, t, dx, dy, p3):
        n_pos = [-dy, dx]
        mag_pos = np.linalg.norm(n_pos)
        nu_pos = n_pos/mag_pos
        u_pos = [(nu_pos[0] * t + p3[0]), (nu_pos[1] * t + p3[1])]
        return u_pos

    def is_inside(self, im, pos):
        if not (0 <= pos[0] < im.shape[0]):
            return False
        if not (0 <= pos[1] < im.shape[1]):
            return False
        return True

    def get_pixels_half (self, pos_or_neg, im, dx, dy, p3):
        color_threshold = (int(np.max(im))+int(np.min(im)))/2
        for ts in (range(len(im[0]))):
            u = self.get_normal_vector((pos_or_neg*(ts+(self.step_size))), dx, dy, p3) 
            test_point = round(u[1]),round(u[0])
            if not self.is_inside(im, test_point):
                return None, None
            test = im[test_point[0], test_point[1]] > color_threshold
            if test == False:
                pixels = ts - 1
                break
        # plt.plot(u[0], u[1], 'c.', markersize=12)
        return pixels, (u[0], u[1])


    def get_calculated_diameter(self, im, p1, p2):
        color_threshold = (int(np.max(im))+int(np.min(im)))/2
        diameters = []
        lines = []
        #mask_meas_lines = np.zeros_like(im)
        for n in range(1, self.n+1): 
            t = 1/(self.n+1) 
            p3, dx, dy = self.get_point((t * n), p1, p2)
            test_point = round(p3[1]),round(p3[0])
            if not self.is_inside(im, test_point):
                continue
            true_point = im[test_point[0], test_point[1]] > color_threshold
            if true_point == False:
                continue
            if true_point == True:
                radius_p, cp1 = self.get_pixels_half(1, im, dx, dy, p3)
                radius_n, cp2 = self.get_pixels_half(-1, im, dx, dy, p3)
                if (radius_p != None) and (radius_n != None):
                    max_val = max(radius_p, radius_n)
                    min_val = min(radius_p, radius_n)
                    equal = abs((max_val - min_val)/(max_val + 1e-5))
                    if equal < 0.1:
                        lines.append((cp1,cp2))
                        diameters.append(radius_p+radius_n)
                        # mask_meas_lines = self.mask_measured_lines(im, lines)
                        # plt.plot(p3[0], p3[1], 'r.', markersize=12)
        calculated_diameter = np.array(diameters).mean()
        return calculated_diameter, lines

    def line_to_arrays(self, line):
        return [line[0][0], line[1][0]], [line[0][1], line[1][1]]

    def mask_measured_lines(self, im, lines):
        mask = np.zeros_like(im)
        for p1, p2 in lines:
            if not (p1 == None or p2 == None):
                cv2.line(mask, np.array(p1).astype(np.int32), np.array(p2).astype(np.int32), 1, 1)
        return mask

    def predict(self, im: np.ndarray):
        x, y, popt, pcov = self.get_fited_line_x_y(im)
        _, _, popt_inv, pcov_inv = self.get_fited_line_y_x(im)
        popt_fit, x_fit, y_fit, p1, p2 = self.get_better_fit(x, y, popt, popt_inv, pcov, pcov_inv)
        calculated_diameter, lines = self.get_calculated_diameter(im, p1, p2)
        mask_meas_lines = self.mask_measured_lines(im, lines)
        #for line in lines:
        #    plt.plot(*self.line_to_arrays(line), 'c-')
        return calculated_diameter, mask_meas_lines
 
if __name__ == "__main__":
    import os 

    model = LineFit(10, 0.5)
    dataset_path = "/Users/carmenlopez/dev/diameterY/scratch/dataset_files"
    example_path = os.path.join(dataset_path, "test_0005.npz")
    example = np.load(example_path)
    diameter_pred, mask_meas_lines = model.predict(example["x"])
    print(diameter_pred, example["d"])
