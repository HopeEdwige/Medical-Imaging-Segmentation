import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

def calculer_et_afficher_histogrammes(chemin_image1, chemin_image2):
    """
    Lit deux images NII, calcule leurs histogrammes normalisés et la distance de Wasserstein,
    puis affiche les histogrammes.

    Args:
        chemin_image1 (str): Chemin vers le fichier NII de la première image.
        chemin_image2 (str): Chemin vers le fichier NII de la deuxième image.
    """
    try:
        # Lire les images avec SimpleITK
        image1 = sitk.ReadImage(chemin_image1)
        image2 = sitk.ReadImage(chemin_image2)

        # Convertir les images en tableaux NumPy
        array_image1 = sitk.GetArrayFromImage(image1).flatten()
        array_image2 = sitk.GetArrayFromImage(image2).flatten()

        # Déterminer les bornes communes pour les bins
        min_intensity = min(array_image1.min(), array_image2.min())
        max_intensity = max(array_image1.max(), array_image2.max())

        # Calculer les histogrammes avec les mêmes bins
        bins = np.linspace(min_intensity, max_intensity, 256)
        hist1, _ = np.histogram(array_image1, bins=bins, density=True)
        hist2, _ = np.histogram(array_image2, bins=bins, density=True)

        # Calculer les centres des bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Calculer la distance de Wasserstein
        wasserstein_distance_value = wasserstein_distance(bin_centers, bin_centers, hist1, hist2)

        # Afficher les histogrammes
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(bin_centers, hist1, color='teal')
        plt.title(f'Histogramme Image 1 ({chemin_image1.split("/")[-1]})')
        plt.xlabel('Intensité')
        plt.ylabel('Fréquence normalisée')

        plt.subplot(1, 2, 2)
        plt.plot(bin_centers, hist2, color='teal')
        plt.title(f'Histogramme Image 2 ({chemin_image2.split("/")[-1]})')
        plt.xlabel('Intensité')
        plt.ylabel('Fréquence normalisée')

        plt.suptitle(f'Distance de Wasserstein : {wasserstein_distance_value:.12f}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustement pour le titre principal

        # Sauvegarder le graphique dans un fichier image
        plt.savefig('distance_wasserstein_plot.png')
        print("Le graphique a été sauvegardé sous 'distance_wasserstein_plot.png'.")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == '__main__':
    # Remplacez ici les chemins vers vos fichiers NII FLAIR et T1CE
    #image_path1 = 'chemin/vers/votre/image_FLAIR.nii.gz'
    #image_path2 = 'chemin/vers/votre/image_T1CE.nii.gz'

    #image_path1 = 'C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-001/day_665/FLAIR.nii'
    #image_path2 = 'C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-001/day_665/T1CE.nii'

    image_path1 = 'C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-018/day_481/FLAIR.nii'
    image_path2 = 'C:/Users/hdodo/OneDrive/Documents/M1-IBIOM/Projet-S2/IRMFlairetTIC/Burdenko-GBM-018/day_481/T1CE.nii'

    # Assurez-vous que les fichiers existent aux chemins spécifiés
    try:
        with open(image_path1, 'r'):
            pass
        with open(image_path2, 'r'):
            pass
    except FileNotFoundError:
        print("Erreur : Veuillez vérifier que les chemins des fichiers sont corrects.")
    else:
        calculer_et_afficher_histogrammes(image_path1, image_path2)