import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import zoom

def load_nii_image(file_path):
    """Charge une image .nii et retourne les données et l'objet image."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return data, img
    except FileNotFoundError:
        print(f"Erreur: Le fichier à l'emplacement '{file_path}' n'a pas été trouvé.")
        return None, None

def plot_histogram(data, title="Histogramme de l'image", bins=100):
    """Affiche l'histogramme des intensités de l'image."""
    if data is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(data.flatten(), bins=bins, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel("Intensité")
        plt.ylabel("Fréquence")
        plt.grid(True)
        plt.show()

def apply_thresholding(data, threshold):
    """Applique un seuillage simple à l'image.
    Les voxels avec une intensité supérieure au seuil sont mis à 1, sinon à 0."""
    if data is not None:
        segmented_data = np.zeros_like(data)
        segmented_data[data > threshold] = 1
        return segmented_data
    return None


#def downsample_data(data, scale_factor=0.5):
#    """Réduit la résolution des données en utilisant un facteur d'échelle."""
#    return zoom(data, scale_factor)

def sample_voxels(data, sample_size=100000):
    """Échantillonne un sous-ensemble aléatoire de voxels."""
    flattened_data = data.flatten()
    if len(flattened_data) > sample_size:
        indices = np.random.choice(len(flattened_data), sample_size, replace=False)
        return flattened_data[indices].reshape(-1, 1)
    return flattened_data.reshape(-1, 1)


def apply_kmeans_clustering_sampled(data, sampled_data, n_clusters=2):
    """Applique K-means sur des voxels échantillonnés et mappe les résultats à la forme originale."""
    if data is not None and sampled_data is not None:
        # Appliquer K-means sur les données échantillonnées
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        sampled_clusters = kmeans.fit_predict(sampled_data)

        # Créer un tableau pour les clusters avec la même forme que l'image originale
        clustered_data = np.zeros(data.shape, dtype=np.int32)

        # Obtenir les indices des voxels échantillonnés
        flattened_data = data.flatten()
        indices = np.random.choice(len(flattened_data), len(sampled_data), replace=False)

        # Mapper les clusters échantillonnés aux indices correspondants
        clustered_data_flat = clustered_data.flatten()
        clustered_data_flat[indices] = sampled_clusters
        return clustered_data_flat.reshape(data.shape)
    return None

def save_segmented_nii(segmented_data, original_img, output_path="test3/Burdenko-GBM-001/segmented_image.nii.gz"):
    """Sauvegarde les données segmentées dans un nouveau fichier .nii.gz."""
    if segmented_data is not None and original_img is not None:
        segmented_img = nib.Nifti1Image(segmented_data, original_img.affine, original_img.header)
        nib.save(segmented_img, output_path)
        print(f"Image segmentée sauvegardée à : {output_path}")

if __name__ == "__main__":
    # Remplacez par le chemin d'accès à votre fichier .nii FLAIR
    flair_file_path = "C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-001/day_063/FLAIR.nii.gz"
    flair_data, flair_img = load_nii_image(flair_file_path)

    if flair_data is not None:
        plot_histogram(flair_data, title="Histogramme de l'image FLAIR")

        # *** SEUIL : À DÉTERMINER À PARTIR DE L'HISTOGRAMME FLAIR ***
        threshold_flair = 100  # Exemple de seuil, À MODIFIER

        segmented_flair_thresholded = apply_thresholding(flair_data, threshold_flair)
        save_segmented_nii(segmented_flair_thresholded, flair_img, "segmented_flair_thresholded.nii.gz")

        # *** NOMBRE DE CLUSTERS POUR K-MEANS : À DÉTERMINER (par exemple, basé sur le nombre de pics) ***
        n_clusters_flair = 3  # Exemple de nombre de clusters, À MODIFIER

        #downsampled_flair_data = downsample_data(flair_data, scale_factor=0.5)
        #segmented_flair_kmeans = apply_kmeans_clustering(downsampled_flair_data, n_clusters_flair)
        
        sampled_data = sample_voxels(flair_data)
        segmented_flair_kmeans = apply_kmeans_clustering_sampled(flair_data, sampled_data, n_clusters_flair)
        save_segmented_nii(segmented_flair_kmeans, flair_img, "segmented_flair_kmeans.nii.gz")

    # Remplacez par le chemin d'accès à votre fichier .nii T1CE
    t1ce_file_path = "C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-001/day_063/T1CE.nii.gz"
    t1ce_data, t1ce_img = load_nii_image(t1ce_file_path)

    if t1ce_data is not None:
        plot_histogram(t1ce_data, title="Histogramme de l'image T1CE")

        # *** SEUIL : À DÉTERMINER À PARTIR DE L'HISTOGRAMME T1CE ***
        threshold_t1ce = 150  # Exemple de seuil, À MODIFIER

        segmented_t1ce_thresholded = apply_thresholding(t1ce_data, threshold_t1ce)
        save_segmented_nii(segmented_t1ce_thresholded, t1ce_img, "segmented_t1ce_thresholded.nii.gz")

        # *** NOMBRE DE CLUSTERS POUR K-MEANS : À DÉTERMINER ***
        n_clusters_t1ce = 3  # Exemple de nombre de clusters, À MODIFIER

        sampled_t1ce_data = sample_voxels(t1ce_data)
        segmented_t1ce_kmeans = apply_kmeans_clustering_sampled(t1ce_data, sampled_t1ce_data, n_clusters_t1ce)
        save_segmented_nii(segmented_t1ce_kmeans, t1ce_img, "segmented_t1ce_kmeans.nii.gz")

    print("Terminé.")