import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import nibabel as nib

def calculer_histogramme_nii(image_path):
    """Calcule l'histogramme d'une image .nii."""
    img = nib.load(image_path)
    data = img.get_fdata()
    # Aplatir les données et calculer l'histogramme
    return np.histogram(data.flatten(), bins=256, density=True)[0]

def calculer_distance_wasserstein_nii(image_path1, image_path2):
    """Calcule la distance de Wasserstein entre deux images .nii."""
    histogramme1 = calculer_histogramme_nii(image_path1)
    histogramme2 = calculer_histogramme_nii(image_path2)
    return wasserstein_distance(np.arange(256), np.arange(256), histogramme1, histogramme2)

# Chemins vers vos images .nii
"""
image_path1 = 'C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-001/day_665/FLAIR.nii'
image_path2 = 'C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-001/day_665/T1CE.nii'
"""
image_path1 = 'C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-018/day_481/FLAIR.nii'
image_path2 = 'C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-018/day_481/T1CE.nii'

# Vérifier si les fichiers existent
# Calculer et afficher la distance de Wasserstein
distance = calculer_distance_wasserstein_nii(image_path1, image_path2)
print(f"Distance de Wasserstein : {distance}")

# Afficher les histogrammes
histogramme1 = calculer_histogramme_nii(image_path1)
histogramme2 = calculer_histogramme_nii(image_path2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(histogramme1)
plt.title('Histogramme Image 1 (FLAIR.nii)')
plt.subplot(1, 2, 2)
plt.plot(histogramme2)
plt.title('Histogramme Image 2 (T1CE.nii)')

# Ajouter la distance de Wasserstein au titre global
plt.suptitle(f"Distance de Wasserstein : {distance}")

# Sauvegarder le graphique dans un fichier image
plt.savefig('distance_wasserstein_plot.png')

# Afficher le graphique
plt.show()