# PSI-PROJEKT
Detekcja Obiektów — Faster R-CNN z PyTorch
Projekt realizuje detekcję obiektów przy użyciu modelu Faster R-CNN z biblioteką PyTorch i adnotacjami w formacie COCO. System umożliwia trening, ewaluację, predykcję i porównanie wyników wielu eksperymentów w sposób zautomatyzowany.

Struktura projektu:
- dataloader.py – DataLoader z COCO i transformacjami
- download_data.py – Pobieranie i rozpakowanie danych z Google Drive
- preprocess.py – Podział na train/val/test i zapis plików .json
- model.py – Konfiguracja modelu Faster R-CNN
- train.py – Trening + ewaluacja z logowaniem wyników
- test.py – Obliczanie mAP, zapisywanie wyników i wykresów
- infernce.py – Pipeline do predykcji i wizualizacji ramek
- run_all_experiments.py – Główny plik do uruchamiania wszystkich eksperymentów
- requirements.txt – Lista zależności
