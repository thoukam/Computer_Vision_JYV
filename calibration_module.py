import cv2
import numpy as np
import os

def calibrate_camera(image_dir, pattern_size=(9, 6), display=False):
    """
    Calibre une caméra à partir d'un dossier contenant des images d'un échiquier.

    Args:
        image_dir (str): Dossier contenant les images.
        pattern_size (tuple): Taille de la grille interne de l’échiquier (cols, rows).
        display (bool): Affiche les coins détectés si True.

    Returns:
        dict: Contenant 'K', 'dist', 'rvecs', 'tvecs', 'reprojection_error'.
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    image_shape = None

    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, filename)
            img = cv2.imread(image_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_shape = gray.shape[::-1]

            ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            if ret:
                objpoints.append(objp)
                corners_subpix = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                imgpoints.append(corners_subpix)

                if display:
                    cv2.drawChessboardCorners(img, (cols, rows), corners_subpix, ret)
                    cv2.imshow("Coins détectés", img)
                    cv2.waitKey(300)

    cv2.destroyAllWindows()

    if len(objpoints) < 3:
        raise ValueError("Moins de 3 images valides pour la calibration.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    reprojection_error = compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)

    return {
        "K": K,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "reprojection_error": reprojection_error
    }

def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    return total_error / len(objpoints)

def save_calibration(filename, data):
    """
    Sauvegarde les paramètres de calibration dans un fichier .npz.

    Args:
        filename (str): Nom du fichier à sauvegarder.
        data (dict): Données à sauvegarder.
    """
    np.savez(filename, K=data["K"], dist=data["dist"], rvecs=data["rvecs"], tvecs=data["tvecs"])