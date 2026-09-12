#  RootSprout-Deep learning platform



RootSprout is a software platform for dynamic root phenotyping during early seed germination. It combines YOLO object detection and UNet pixel-wise segmentation to automatically identify germinated and non-germinated seeds, track seed regions, and extract traits such as radicle length, root area, root perimeter, 2DCI, and germination rate. The results are saved as CSV files and image sequences.


Yu-Peng Zhu (朱玉鹏) <sup>1 </sup>, Peng Wang(汪鹏)<sup>1*,*4 </sup>

<sup>1</sup>Nanjing Agricultural
University, State Key Laboratory of Crop Genetics and Germplasm Enhancement,
College of Resources and Environmental Sciences, Nanjing, Jiangsu 210095, China*


<sup>4</sup>Center for Agriculture
and Health, Academy for Advanced Interdisciplinary Studies, Nanjing
Agricultural University, Nanjing 210095, China*

<sup> * </sup>Correspondence for the source code:Yu-Peng Zhu([2023203050@stu.njau.edu.cn](mailto:2018101176@njau.edu.cn)) and Peng Wang ([p.wang3@naju.edu.cn](mailto:p.wang3@naju.edu.cn)) 


- Project page: https://github.com/The-Wang-Lab-NAU/RootSprout
- Downloads (Releases): https://github.com/The-Wang-Lab-NAU/RootSprout/releases

---

## 1. What RootSprout Can Do

- Batch process time-series images
- Automatically detect germinated and non-germinated seed regions
- Track seeds across frames and reconstruct movement trajectories
- Segment seeds, roots, and seedlings
- Extract static seed traits and dynamic radicle traits
- Export CSV tables and cropped image sequences
- Provide a graphical user interface (GUI) that requires no programming

---

## 2. System Requirements

### Recommended Configuration

- OS: Windows 10 / 11, 64-bit
- RAM: 8 GB or more, 16 GB recommended
- Hard disk: at least 5 GB free space
- Optional GPU: NVIDIA GPU with CUDA for accelerated deep learning inference
- No Python installation required if you use the provided `.exe`

### Source Code Environment

- Python 3.8–3.10
- PyQt5 5.15.10
- PyTorch 1.13.1
- OpenCV 4.7.0.72
- Other dependencies are listed in Section 4

---

## 3. Download and Installation

### 3.1 Download Weight Files and Test Images

1. Open the Releases page:  
   https://github.com/The-Wang-Lab-NAU/RootSprout/releases

2. Find the **Assets** section under the latest release.

3. Download the following files:
   - **Weight files**: model weights (YOLO and UNet)
   - **Test_image**: example image sequences for quick testing

4. After unzipping the weight files, place the following files into the **`model_data/`** directory:

```text
RootSprout/
├── model_data/
│   ├── voc_classes.txt
│   ├── Weights_UNET.pth
│   ├── Weights_yolo.pth
│   └── yolo_anchors.txt
└── ...
```

5. The example images (`Test_image`) can be placed in any path containing only English characters, e.g., `D:\RootSprout_Test\`.

### 3.2 Obtain the `.exe` Executable

Because the `.exe` file is too large to upload directly to GitHub, please send an email to:

**2023203050@stu.njau.edu.cn**

In your email, please include:
- Your name
- Your institution / university
- Purpose of use
- Operating system version (e.g., Windows 10/11)

We will reply as soon as possible with a download link.

### 3.3 Run from Source (for Developers)

1. Install Python 3.8–3.10. Miniconda or Anaconda is recommended.

2. Clone the repository:

```bash
git clone https://github.com/The-Wang-Lab-NAU/RootSprout.git
cd RootSprout
```

3. Create and activate a conda environment:

```bash
conda create -n rootsprout python=3.9
conda activate rootsprout
```

4. Install dependencies (pip is recommended for exact versions):

```bash
pip install torch==1.13.1 torchvision==0.14.1
pip install opencv-python==4.7.0.72
pip install scikit-learn==1.2.1
pip install scikit-image==0.19.3
pip install matplotlib==3.7.0
pip install pandas==1.5.3
pip install numpy==1.23.5
pip install scipy==1.10.0
pip install seaborn==0.12.2
pip install PyQt5==5.15.10
```

Alternatively, you can try conda:

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 -c pytorch
conda install opencv scikit-learn scikit-image matplotlib pandas numpy scipy seaborn pyqt=5.15.10 -c conda-forge
```

5. Download the weight files (see 3.1) and place them into the `model_data/` directory.

6. Run the main program:

```bash
python RootScrout_GUI.py
```

---

## 4. Repository Structure

```text
RootSprout/
├── Time-series data_sorted/       # Time-series image data (example/test input)
├── img_crop/                      # Cropped images or intermediate results
├── model_data/                    # Model weights
│   ├── voc_classes.txt
│   ├── Weights_UNET.pth
│   ├── Weights_yolo.pth
│   └── yolo_anchors.txt
├── nets/                          # Network definitions (YOLO/UNet architectures)
├── utils/                         # General utility functions
├── utils_unet/                    # UNet-related utility functions
├── Image processing pipeline.ipynb # Image processing pipeline notebook
├── LICENSE.txt                    # License
├── README.md                      # Documentation
├── RootScrout_GUI.py              # GUI main entry
├── unet.py                        # UNet segmentation module
└── yolo.py                        # YOLO detection module
```

---

## 5. Dependency List

| Library | Version |
|---------|---------|
| torch | 1.13.1 |
| opencv-python | 4.7.0.72 |
| scikit-learn | 1.2.1 |
| scikit-image | 0.19.3 |
| matplotlib | 3.7.0 |
| pandas | 1.5.3 |
| numpy | 1.23.5 |
| scipy | 1.10.0 |
| seaborn | 0.12.2 |
| PyQt5 | 5.15.10 |

Other dependencies such as Pillow and torchvision can be installed as needed.

---

## 6. How to Use the Software

After launching the GUI, follow Step 1–Step 5:

1. **Step 1: Select Input Folder**  
   Choose the folder containing time-series images (e.g., `Time-series data_sorted`).  
   Supported formats: `.jpg`, `.png`, `.jpeg`, `.tif`.

2. **Step 2: Select Output Folder**  
   Choose where to save the results.

3. **Step 3: Set Scale**  
   Enter the `mm/pixel` value, e.g., `0.216541`.  
   This parameter converts pixels to real-world length.

4. **Step 4: Click Start Analysis**  
   The software will automatically detect, track, segment, and extract traits.  
   The log panel on the right shows the progress.

5. **Step 5: Click View Traits Data**  
   View the output CSV tables.  
   Double-click a row to view the corresponding time-series cropped images for that seed.

---

## 7. Output Results

After processing, the output folder will contain files similar to:

```text
Output_Results/
├── Seed_Traits.csv
├── Bounding_Box_Aspect_Ratio_Traits.csv
├── Bounding_Box_Area_Traits.csv
├── Bounding_Box_Perimeter_Traits.csv
├── Bounding_Box_Track_Length.csv
├── Root_Length.csv
├── Root_Diameter.csv
├── Root_Area.csv
├── Root_Perimeter.csv
├── Germination_Rate.csv
├── Root_2DCI.csv
└── Region_Crop/
```

In some versions, file names may be `Radicle_Length.csv`, `Radicle_2DCI.csv`, etc. Please refer to the actual output.

---

## 8. FAQ

### 1) The exe does not start or crashes immediately?

- Make sure you have extracted all files completely; do not run it directly from the compressed archive.
- Ensure the path contains no Chinese characters or spaces.
- Install Microsoft Visual C++ Redistributable.
- Check whether antivirus software is blocking it.

### 2) Error: `no Qt platform plugin`?

- If running from source, reinstall PyQt5:

```bash
pip uninstall PyQt5
pip install PyQt5==5.15.10
```

- If using the exe, please make sure the Qt plugins were included during packaging.

### 3) Error: `YOLO and Unet model files are not found`?

Make sure `yolo.py`, `unet.py`, and the main program are in the same directory, or that they are correctly packaged. If the code references `nets`, `utils`, etc., ensure these directories are complete.

### 4) Model weights not found?

Download the weights from the **Weight files** section on the Releases page. After unzipping, place them into `model_data/` and ensure the following files exist:

```text
model_data/voc_classes.txt
model_data/Weights_UNET.pth
model_data/Weights_yolo.pth
model_data/yolo_anchors.txt
```

### 5) Processing is slow?

- Use an NVIDIA GPU and install the PyTorch version matching your CUDA.
- Reduce the number of images or their resolution.
- Close other programs that occupy GPU memory.

### 6) How to package the `.exe` myself?

Install PyInstaller:

```bash
pip install pyinstaller
```

Example packaging command:

```bash
pyinstaller --noconfirm --onedir --windowed --name RootScrout ^
  --add-data "yolo.py;." ^
  --add-data "unet.py;." ^
  --add-data "nets;nets" ^
  --add-data "utils;utils" ^
  --add-data "utils_unet;utils_unet" ^
  --add-data "model_data;model_data" ^
  RootScrout_GUI.py
```

The generated exe will be located at:

```text
dist/RootScrout/RootScrout.exe
```

---

## 9. Citation

If you use RootSprout in your research, please cite our paper:

```bibtex
@article{RootSprout2026,
  title   = {RootSprout: A deep learning-powered phenotyping platform for rapid profiling arsenic tolerance in germinating rice},
  
}
```

Please update this with the final publication details.

---

## 10. Contact and Copyright

- Laboratory: Environmental Biology Laboratory, Nanjing Agricultural University
- GitHub: https://github.com/The-Wang-Lab-NAU/RootSprout
- Email: 2023203050@stu.njau.edu.cn
- Copyright: Copyright 2026, Environmental Biology Laboratory, Nanjing Agricultural University, Nanjing, China

If you have any questions, please open an issue on GitHub or send an email to the address above.

