import numpy as np
import matplotlib.pyplot as plt

def normal_map_from_depth_map(depthmap):
    h, w = np.shape(depthmap)
    normals = np.zeros((h, w, 3))
    for x in range(1, h - 1):
        for y in range(1, w - 1):
            dzdx = (float((depthmap[x + 1, y])) - float((depthmap[x - 1, y]))) / 2.0
            dzdy = (float((depthmap[x, y + 1])) - float((depthmap[x, y - 1]))) / 2.0

            n = np.array([-dzdx, -dzdy, 1.0])

            n = n * 1/np.linalg.norm(n)

            normals[x, y] = n * 0.5 + 0.5
    normals *= 255
    normals = normals.astype('uint8')
    plt.imshow(normals)
    plt.show()
    #normals = cv2.cvtColor(normals, cv2.COLOR_BGR2RGB)

    #cv2.imwrite("images/N_{0}".format(fileName), normals)
