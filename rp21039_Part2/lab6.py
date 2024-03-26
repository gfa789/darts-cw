'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Lab Sheet 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math

'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

# print("here", flush=True)
if __name__ == '__main__': 
    bDisplayAxis = True

    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    name_list.append('sphere_r')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=2)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, -4],
                     [0, 1, 0, 2],
                     [0, 0, 1, -2],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])

    name_list.append('sphere_g')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=2)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, -7],
                     [0, 1, 0, 2],
                     [0, 0, 1, 3],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])

    name_list.append('sphere_b')
    sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    mesh_list.append(sph_mesh)
    H_list.append(np.array(
                    [[1, 0, 0, 4],
                     [0, 1, 0, 1.5],
                     [0, 0, 1, 4],
                     [0, 0, 0, 1]]
            ))
    RGB_list.append([0., 0.5, 0.5])

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if bDisplayAxis:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * 45*5/180.
    # theta = 0.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * 80/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * 45*5/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)
    render_list = [(H0_wc, 'view0.png', 'depth0.png'), 
                   (H1_wc, 'view1.png', 'depth1.png')]


    
    #########################################
    '''
    Question 1: Epipolar line
    Hint: check reference here
    http://www.open3d.org/docs/0.7.0/tutorial/Basic/visualization.html#draw-line-set

    Write your code here
    '''
    # first need to find camera 0 centre (COP) in world coordinates
    # Note Pc'=H0_wc*Pw' using homogeneous coordinates, hence Pc=R0_wc*Pw+T0_wc 
    # and Pw=inv(R0_wc)(Pc-T0_wc) using 3-D coordinates
    # Camera centre in 3-D camera coordinates is Pc=(0,0,0), hence camera centre
    # in world coordinates is Pw=-inv(R0_wc)*T0_wc. We'll call this pt0:
    pt0 = -np.matmul(np.linalg.inv(H0_wc[:3,:3]),H0_wc[:3,3])
    
    # the other point is the centre of one of the spheres
    # we'll use the last one and its position is given by 
    # last column of coordinate transformation matrix (index -1), ie
    pt1 = H_list[-1][:3, 3]   # the 3-D vector is the first 3 elements
    
    # these define the end points of the line                           
    end_pts = [pt0[:], pt1[:]]
    # now use open3D LineSet to create line
    lines = [[0, 1]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(end_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    obj_meshes.append(line_set)
    #########################################


    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.
    
    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width=640
    img_height=480
    f=415 # focal length
    # image centre in pixel coordinates
    ox=img_width/2-0.5 
    oy=img_height/2-0.5
    K = o3d.camera.PinholeCameraIntrinsic(img_width,img_height,f,f,ox,oy)

    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()


    #########################################
    '''
    Question 2: Extend epipolar line in the image
    
    Write your code here
    '''
    # To show the complete projected ray in the camera 1 image, which corresponds
    # to the epipolar line, we project the two 3-D points known to lie on the ray
    # into the image and then draw the line in the image that intersects both
    # projected points
    
    # stack 3-D line points and convert to camera 1 coordinates
    pts_w = np.stack((pt0, pt1))
    pts_cam1 = transform_points(pts_w, H1_wc)
    
    # define M matrix to convert pixel coordinates to image coordinates
    # as per slide 5, lec 3
    # Note that pixel sizes (sx,sy) in Open3D are both 1
    sx=sy=1
    Mpi=np.array([[sx, 0, -ox/f],[0, sy, -oy/f],[0, 0, 1]])
    # now compute matrix which converts iamge coordinates to pixel coordinates
    Mip=np.linalg.inv(Mpi)
    
    # project both points into camera 1 image plane and convert to pixel coordinates
    pts_img1 = []
    for i in range(pts_cam1.shape[0]):
        img_pt = f*pts_cam1[i,:]/pts_cam1[i,2]
        img_pt = np.matmul(Mip,img_pt[:].reshape(3,1)).reshape(3,)
        pts_img1.append(img_pt[:2])
        
    # find the pixel coordinates of the line on left edge (x pixel coordinate = 0)
    # and top edge (y pixel coordinate = 0) and convert to integers for cv2.line()
    cam0_centre = pts_img1[0]
    sphere_centre = pts_img1[1]
    slope=(cam0_centre[1]-sphere_centre[1])/(cam0_centre[0]-sphere_centre[0])
    left_y = sphere_centre[1] + ((0-sphere_centre[0]) * slope)
    left_end_pt = np.array([0, left_y]).astype(int)
    right_x = -left_y/slope
    right_end_pt = np.array([right_x, 0]).astype(int)
    
    # draw the line in the image
    img = cv2.imread('view1.png')
    img = cv2.circle(img, sphere_centre.astype(int), radius=0, color=(0, 0, 255), thickness=4)
    img = cv2.circle(img, left_end_pt, radius=0, color=(255, 0, 0), thickness=4)
    img = cv2.circle(img, right_end_pt, radius=0, color=(0, 255, 0), thickness=4)
    img = cv2.line(img, left_end_pt, right_end_pt, (0,255,0), 1)
    cv2.imwrite('view1_eline_extend.png', img)
    #########################################


    #########################################
    '''
    Question 3: Draw epipolar line using essential and fundamental matrices

    Write your code here
    '''

    # Now we draw the line using the essential and fundamental matrices
    
    # compute essential and fundamental matrix
    # first get coordinate transformation from cam1 to cam0
    H_10 = np.matmul(H0_wc, np.linalg.inv(H1_wc))
    
    # this gives P0=R10*P1+T10
    # hence P1=R10^T(P0-T10) which is in the same the form as slide 3, lec 2
    # hence
    R = H_10[:3, :3].T # R10^T
    T = H_10[:3, 3] # T10
    
    # now form the cross pdt matrix S from T - slide 9, lec 2
    S = np.array([
        [0, -T[2], T[1]],
        [T[2], 0, -T[0]],
        [-T[1], T[0], 0]
    ])
    
    # and hence the essential matrix as per slide 11, lec 2
    E = np.matmul(R, S)
    print('Essential Matrix')
    print(E)
    
    # now use the pixel to image coordinate matrix to compute the 
    # fundamental matrix as per slide 5, lec 3
    F = np.matmul(np.matmul(Mpi.T, E), Mpi)
    print('Fundamental Matrix:')
    print(F)

    # get an image point in camera 0 for centre of last sphere
    pt1_cam0 = transform_points(pt1.reshape(1,3), H0_wc).reshape(3,)
    pt1_img0 = f*pt1_cam0[:]/pt1_cam0[2]
    pt1_img0 = np.matmul(Mip,pt1_img0[:].reshape(3,1))

    # compute epipolar line as per slides 3 and 5, lec 3
    u = np.matmul(F,pt1_img0).reshape(3,)

    # hence dot product of u with image point in other camaera 
    # then defines the epipolar line, ie dot(pt1_img1,u)=0
    # hence we can get the end points of the line on the left (x coordinate = 0)
    # and on the right (x coordinate = img_width). Then convert to integers for cv2.line()
    p0 = np.array([0, -f*u[2]/u[1] ]).astype(int)
    p1 = np.array([img_width, -(f*u[2]+u[0]*img_width)/u[1] ]).astype(int)

    # visualise
    img = cv2.imread('view1_eline_extend.png')
    img = cv2.line(img, p0, p1, (255,0,0), 1)
    cv2.imwrite('view1_eline_fmat.png', img)
    #########################################
