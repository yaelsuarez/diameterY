import cv2
import numpy as np

img_size = [224, 224]


class Fiber:
    def __init__(self, p1, p2, diameter, brightness: int):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.diameter = diameter
        self.brightness = brightness
        self.u_normal = self.get_unit_normal_vector()

    def get_full_mask(self):
        mask = np.zeros(img_size, dtype=np.uint8)
        cv2.line(mask, self.p1, self.p2, 1, self.diameter)
        return mask.astype(np.bool)

    def get_center_mask(self):
        mask = np.zeros(img_size, dtype=np.uint8)
        cv2.line(mask, self.p1, self.p2, 1, 2)
        return mask.astype(np.bool)

    def get_unit_vector(self):
        d = self.p2 - self.p1
        magnitude = np.linalg.norm(d)
        return d / magnitude

    def get_unit_normal_vector(self):
        u = self.get_unit_vector()
        transform_matrix = np.array([[0, 1], [-1, 0]])
        return np.matmul(u, transform_matrix)

    def get_halves(self):
        translation = self.u_normal * self.diameter / 4
        d = round(self.diameter / 2)
        # positive mask
        p1_p = np.round(self.p1 + translation).astype(np.int32)
        p2_p = np.round(self.p2 + translation).astype(np.int32)
        mask_p = np.zeros(img_size, dtype=np.uint8)
        cv2.line(mask_p, p1_p, p2_p, 1, d)
        # negative mask
        p1_n = np.round(self.p1 - translation).astype(np.int32)
        p2_n = np.round(self.p2 - translation).astype(np.int32)
        mask_n = np.zeros(img_size, dtype=np.uint8)
        cv2.line(mask_n, p1_n, p2_n, 1, d)
        return mask_p.astype(np.bool), mask_n.astype(np.bool)

    def get_vector_field(self):
        field = np.zeros(
            img_size
            + [
                2,
            ],
            dtype=np.float32,
        )
        mask_p, mask_n = self.get_halves()
        mask_c = self.get_center_mask()
        u_normal = self.u_normal * [-1, 1]
        field[mask_p] = u_normal
        field[mask_n] = -u_normal
        field[mask_c] = 0
        return field

    @classmethod
    def create(cls):
        direction = np.random.rand() > 0.5
        width, height = img_size
        if direction:
            x1, y1 = np.random.randint(0, width + 1), 0
            x2, y2 = np.random.randint(0, width + 1), height
        else:
            x1, y1 = 0, np.random.randint(0, height + 1)
            x2, y2 = width, np.random.randint(0, height + 1)
        p1, p2 = np.array((x1, y1)), np.array((x2, y2))
        brightness = np.random.randint(120, 255)
        diameter = np.random.randint(round(width / 20), round(width / 4))
        return cls(p1, p2, diameter, brightness)


class Image:
    def __init__(self, fiber: Fiber, bg_color: int):
        self.fiber = fiber
        self.bg_color = bg_color

    def render_image(self):
        im = np.full(img_size, self.bg_color, dtype=np.uint8)
        mask = self.fiber.get_full_mask()
        im[mask] = self.fiber.brightness
        return im

    def render_field_and_weights(self):

        field = self.fiber.get_vector_field()
        mask = self.fiber.get_full_mask()
        
        Rcsb = img_size[0] * img_size[1]
        Rcs = np.count_nonzero(mask)
        Rb = Rcsb - Rcs
        Wcs = Rb/Rcsb
        Wb = Rcs/Rcsb

        w = np.zeros(img_size, dtype=np.float32)
        w[mask] = Wcs
        w[~mask] = Wb
        w = np.expand_dims(w, axis=-1)
        
        f = np.concatenate((field, w), axis=-1)

        return f


    @classmethod
    def create(cls):
        color_background = np.random.randint(40, 80)
        fiber = Fiber.create()
        return cls(fiber, color_background)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = Image.create()
    f = img.render_field_and_weights()
    print(f.shape)
    plt.imshow(f[..., 2])
    plt.show()
