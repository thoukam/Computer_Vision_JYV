import cv2
import numpy as np
import open3d as o3d
import os

def run_processing(left_image, right_image, K, dist, log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    # Étape 1 : Détection et description avec SIFT
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(left_image, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(right_image, None)

    # Étape 2 : Appariement avec BFMatcher + ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    log(f"[INFO] Bonnes correspondances : {len(good_matches)}")

    if len(good_matches) < 8:
        log("[ERREUR] Trop peu de correspondances fiables pour continuer.")
        return

    # Étape 3 : Estimation de la matrice fondamentale
    pts1 = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    inliers = mask.ravel()
    log("[INFO] Matrice fondamentale F:\n" + str(F))

    # Étape 4 : Estimation de la matrice essentielle et récupération de la pose
    E = K.T @ F @ K
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    log("[INFO] Matrice essentielle E:\n" + str(E))
    log("[INFO] Rotation R:\n" + str(R))
    log("[INFO] Translation t:\n" + str(t))

    # Étape 5 : Triangulation
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    points_4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3D = (points_4D[:3] / points_4D[3]).T

    points_3D = points_3D[points_3D[:, 2] > 0]  # Ne garder que les points devant la caméra
    log(f"[INFO] Points 3D reconstruits : {points_3D.shape[0]}")

    # Étape 6 : Visualisation ou export du nuage
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D)
    colors = np.random.rand(points_3D.shape[0], 3)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    ply_filename = "reconstruction.ply"
    o3d.io.write_point_cloud(ply_filename, pcd)
    log(f"[INFO] Nuage de points sauvegardé dans : {ply_filename}")

    if os.environ.get("DISPLAY"):
        try:
            o3d.visualization.draw_geometries([pcd])
        except Exception as e:
            log(f"[AVERTISSEMENT] Échec de l'affichage : {str(e)}")
    else:
        log("[INFO] Affichage désactivé : aucune session graphique détectée (DISPLAY manquant).")