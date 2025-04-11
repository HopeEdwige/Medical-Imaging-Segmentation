import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def load_nii_image(file_path):
    """Charge une image .nii et retourne les données et l'objet image."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return data, img
    except FileNotFoundError:
        print(f"Erreur: Le fichier à l'emplacement '{file_path}' n'a pas été trouvé.")
        return None, None

def apply_thresholding(data, threshold):
    """Applique un seuillage simple à l'image.
    Les voxels avec une intensité supérieure au seuil sont mis à 1, sinon à 0."""
    if data is not None:
        segmented_data = np.zeros_like(data)
        segmented_data[data > threshold] = 1
        return segmented_data
    return None

def save_segmented_nii(segmented_data, original_img, output_path="segmented_image.nii.gz"):
    """Sauvegarde les données segmentées dans un nouveau fichier .nii.gz."""
    if segmented_data is not None and original_img is not None:
        segmented_img = nib.Nifti1Image(segmented_data, original_img.affine, original_img.header)
        nib.save(segmented_img, output_path)
        print(f"Image segmentée sauvegardée à : {output_path}")

if __name__ == "__main__":
    # Remplacez par les chemins d'accès à vos fichiers .nii.gz
    flair_file_path = "C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-001/day_665/FLAIR.nii.gz"
    t1ce_file_path = "C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-001/day_665/T1CE.nii.gz"

    # Chargement des images FLAIR
    flair_data, flair_img = load_nii_image(flair_file_path)
    if flair_data is not None:
        # Seuil proposé pour FLAIR
        threshold_flair = 100
        segmented_flair = apply_thresholding(flair_data, threshold_flair)
        save_segmented_nii(segmented_flair, flair_img, "segmented_flair.nii.gz")

    # Chargement des images T1CE
    t1ce_data, t1ce_img = load_nii_image(t1ce_file_path)
    if t1ce_data is not None:
        # Seuil proposé pour T1CE
        threshold_t1ce = 150
        segmented_t1ce = apply_thresholding(t1ce_data, threshold_t1ce)
        save_segmented_nii(segmented_t1ce, t1ce_img, "segmented_t1ce.nii.gz")

    print("Segmentation terminée.")