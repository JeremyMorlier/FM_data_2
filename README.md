## Installation
```bash
python3 -m venv venv_mobile_clip  
source venv_mobile_clip/bin/activate
pip install ftfy regex tqdm
pip install third_party/CLIP  
pip install -r third_party/dinov2/requirements.txt -r third_party/dinov2/requirements-extras.txt  
pip install -e third_party/segment-anything/I
```

```bash
python3 -m venv venv_mobile_clip  
source venv_mobile_clip/bin/activate
pip install -r requirements.txt
pip install xformers
pip install third_party/CLIP
pip install third_party/segment-anything
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com cuml-cu11
```


## Test Installation
```bash
python3 installation_test/clip_test.py
python3 installation_test/sam_test.py
```