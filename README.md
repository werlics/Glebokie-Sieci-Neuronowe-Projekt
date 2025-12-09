# Glebokie-Sieci-Neuronowe-Projekt
Klasyfikacja dźwięków otoczenia z wykorzystaniem głębokich sieci neuronowych na podstawie plików audio
## Dataset UrbanSound8K
https://www.kaggle.com/datasets/chrisfilo/urbansound8k/data
## Dependencies 
Dla kart graficznych z CUDA 13.0 by wszystko odpowiednie działało (wersja pytorch 12.8 bo na nowszych nie działa FFmpeg, przez co torchaudio.load() nie wczytuje plików audio)
```
pip install -r req.txt
```
