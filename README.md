# SAM Local Demo (Windows)

Du an nay dong goi `Segment Anything` de chay local tren Windows voi 2 kieu demo:

- Demo tuong tac bang `point/box` (`sam_local_app.py`)
- Demo bang `text prompt` ket hop Grounding DINO + SAM (`sam_text_local_app.py`)

Mac dinh du an dang chay tren CPU va dung checkpoint `sam_vit_b_01ec64.pth`.

## 1) Cau truc thu muc

```text
sam_project_2/
|- segment-anything/          # Source goc cua Meta SAM (thu vien chinh)
|- models/                    # Model checkpoint (.pth), vi du: sam_vit_b_01ec64.pth
|- venv/                      # Python virtual environment
|- sam_local_app.py           # Flask app demo point/box (port 7861)
|- sam_text_local_app.py      # Flask app demo text prompt (port 7862)
|- Start_SAM_Demo.bat         # File chay nhanh demo point/box
|- Start_SAM_Text_Demo.bat    # File chay nhanh demo text prompt
|- test_sam_on_image.py       # Script test nhanh tren 1 anh
|- requirements-local.txt     # Danh sach package can cai
|- uploads_gui/               # Anh upload cho demo point/box
|- outputs_gui/               # Ket qua demo point/box
|- uploads_text_gui/          # Anh upload cho demo text prompt
|- outputs_text_gui/          # Ket qua demo text prompt
|- outputs/                   # Ket qua script CLI/thu nghiem
|- .gitignore
```

## 2) Luong hoat dong tong quan

### A. Demo point/box (`sam_local_app.py`)

1. Upload anh
2. Bam `Extract image embedding`
3. Chon prompt:
   - Point duong/am
   - Hoac ve box
4. Bam `Phan vung`
5. App tra ve:
   - Anh goc + prompt
   - Anh mask SAM
   - Anh da tach khoi nen (RGBA)

### B. Demo text prompt (`sam_text_local_app.py`)

1. Upload anh
2. Bam `Extract image embedding`
3. Nhap text prompt (vi du: `dog`, `person`, `red car`)
4. Grounding DINO tim box theo text
5. SAM dung box tot nhat de tao mask
6. App tra ve anh box, anh mask, va anh tach nen

## 3) Cach khoi dong nhanh

### Cach 1: Chay bang file `.bat` (de nhat)

- Demo point/box:

```bat
Start_SAM_Demo.bat
```

Mo trinh duyet tai: `http://127.0.0.1:7861`

- Demo text prompt:

```bat
Start_SAM_Text_Demo.bat
```

Mo trinh duyet tai: `http://127.0.0.1:7862`

### Cach 2: Chay bang lenh Python truc tiep

Tu PowerShell tai thu muc goc du an:

```powershell
cd D:\Python\sam_project_2
.\venv\Scripts\python.exe .\sam_local_app.py
```

Hoac:

```powershell
cd D:\Python\sam_project_2
.\venv\Scripts\python.exe .\sam_text_local_app.py
```

## 4) Cai dat moi truong (neu chua co)

```powershell
cd D:\Python\sam_project_2
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r .\requirements-local.txt
```

## 5) Yeu cau model checkpoint

Can co file:

`models/sam_vit_b_01ec64.pth`

Neu thieu, tai tu link chinh thuc:

- https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

## 6) Thu nhanh bang CLI (khong can giao dien web)

### A. Automatic mask generation (SAM AMG)

```powershell
cd D:\Python\sam_project_2\segment-anything
..\venv\Scripts\python.exe scripts\amg.py --checkpoint "..\models\sam_vit_b_01ec64.pth" --model-type vit_b --device cpu --input "notebooks/images/dog.jpg" --output "..\outputs\demo_amg"
```

Ket qua o:

`outputs/demo_amg/dog/` (gom cac file `*.png` va `metadata.csv`)

### B. Script test 1 diem trung tam

```powershell
cd D:\Python\sam_project_2
.\venv\Scripts\python.exe .\test_sam_on_image.py <duong_dan_anh>
```

## 7) Ghi chu van hanh

- Du an hien tai uu tien CPU, thoi gian suy luan co the cham voi anh lon.
- Neu may co CUDA, co the nang cap de chay nhanh hon (can cai PyTorch ban CUDA).
- Thu muc `segment-anything/` la source goc; phan tuy bien cho app local nam o cac file `sam_*_app.py` tai root.

