import numpy as np

def get_total_flux(flux):
    """Flux has dimensions (height, width, flux)"""
    # N is the binned direction matrix 
    N = np.zeros(flux.shape, dtype=np.int32)
    N[flux > 1/3] = 1
    N[flux < -1/3] = -1
    h, w, _ = N.shape
    total_flux = np.zeros(shape=(h,w))
    for x_position in range(w):
        for y in range(h):
            x = x_position
            visited = [(x,y)]
            dx, dy = N[y,x]
            out_of_bounds = False
            while not(dx == 0 and dy == 0):
                x += dx
                y += dy
                if not 0 <= x < w:
                    out_of_bounds = True
                    break
                if not 0 <= y < h:
                    out_of_bounds = True
                    break
                if (x,y) in visited:
                    #cycle_detected = True
                    break
                dx, dy = N[y,x]
                visited.append((x,y))
            if not out_of_bounds:
                total_flux[y,x] += 1
    return total_flux
