# MandarinMatcher 

**MandarinMatcher** is an image matching program that incorporates researcher verification for the individual identification of Mandarin fish (*Synchiropus splendidus*). 
It employs a ResNet-50 backbone to derive image embeddings and utilizes FAISS for fast nearest-neighbour search. 
This repository contains a streamlined, minimal version of the project developed and evaluated in the associated **Master of Science thesis**. No research pictures, models or personal data are included. Users must submit their own reference and query images. 

## Installation Process 
Create a new virtual environment: 
```bash
python -m venv venv
source venv/bin/activate   
pip install -r requirements.txt

## Features 

- **Automated embedding extraction** using a pretrained ResNet-50 (PyTorch)
- **Fast Top-K similarity search** via FAISS (IndexFlatIP)
- **Human-in-the-loop verification** stored in a CSV log
- **Flask-based web interface** for convenient manual validation
- **Command-line interface** for building and querying the index
- **Reproducible folder structure** for integration into ecological workflows

## Usage
1. Build an index 
python matcher.py build --ref images/reference --outmodel model/
this generates: 
model/faiss.index
model/meta.json

2. Query a new image
python matcher.py query --img images/query/your_image.jpg --model model/ --topk 6
this prints: 
Top-K nearest neighbours
Cosine similarity scores 
and generates:
output/manual_verifications.csv

3. Use the Web Interface (Flask)
python app_flask.py 
open the displayed local URL in your browser 
upload an image -> view the Top-K matches -> select 'match' or 'new'. 
All decisions are saved in: 
output/manual_verifications.csv

## Notes
- FAISS index is not automatically updated to prevent error propagation 
- Curate your accepted matches and rebuild the index intentionally 
- The program runs on CPU only; GPU acceleration is optional but not required 
- Reference images should show left flank of the individual for consistent identification 

## Citation 
if you use this code in research, please cite the associated Master thesis: 
Wilms-Posen, E. (2025). Population decline of Mandarin fish (Synchiropus splendidus) at Banda Naira, Indonesia: automated photographic capture-recapture for estimating abundance. Master's thesis, Bochum, Germany.

## License
This project is licensed under the MIT License, see the LICENSE file for details.


