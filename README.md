# NanoSegment-Microscopy

NanoSegment-Microscopy is a desktop Python application for particle segmentation and quantitative image analysis in AFM, SEM, and TEM micrographs. The current implementation is provided as a Tkinter/Matplotlib GUI in `script.py` and uses Meta AI's Segment Anything Model (SAM) for automatic mask generation.

The software is intended for research workflows where particles are segmented and then used for advanced statistical analysis.

## Repository Layout

```text
NanoSegment-AFM/
├── script.py        # Main Tkinter GUI and CLI quick-test entry point
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies for the full application
├── pyproject.toml           # Minimal project metadata for editable installs
├── LICENSE                  # MIT license
├── .gitignore               # Excludes environments, checkpoints, data, and outputs
└── *.png / *.ico            # Optional GUI icons used by the Tkinter toolbar
```

Optional GUI icon files can be placed next to `script.py`. The code falls back to text labels if they are missing. Icon names used by the script include `BASICSTATISTICALPARAMETERS.png`, `GMM.png`, `Bootstrap.png`, `DBSCAN.png`, `Threshold.png`, `Histogram.png`, `NND.png`, `ttest.png`, `Colormap.png`, `Boxplot.png`, `Bivariate.png`, `Bivariate2.png`, `Overlay.png`, and `button_help.ico`.

## Requirements

- Python 3.10 recommended.
- A desktop environment with Tk support.
- PyTorch installed for CPU or CUDA, depending on the workstation.
- The `segment-anything` Python package.
- A compatible SAM checkpoint file (`.pth`).

On Linux, Tkinter may need to be installed through the system package manager, for example:

```bash
sudo apt-get install python3-tk
```

On Windows and macOS, Tkinter is usually included with the standard Python installers.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/joselorencs/NanoSegmentMicroscopy
cd NanoSegmentMicroscopy
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch. For CPU-only use:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For CUDA-enabled systems, install the PyTorch build recommended for your driver and CUDA version from the official PyTorch installation selector.

Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Alternatively, for an editable local install with the same dependency set:

```bash
python -m pip install -e .
```

## SAM Checkpoints

NanoSegment-Microscopy requires an external SAM checkpoint. The GUI defaults to looking for:

```text
sam_vit_b_01ec64.pth
```

in the current working directory. You can also select any checkpoint through the GUI using the checkpoint file picker.

Common SAM model types supported by the code are:

| Model type | Expected checkpoint family |
| --- | --- |
| `vit_b` | SAM ViT-B / base |
| `vit_l` | SAM ViT-L / large |
| `vit_h` | SAM ViT-H / huge |

The selected `--model-type` or GUI model type must match the checkpoint architecture. Checkpoints are large binary files and should not be committed to Git. The included `.gitignore` excludes `*.pth`, `*.pt`, and related model files.

## Usage

Launch the GUI:

```bash
python script.py
```

Typical GUI workflow:

1. Click `IMPORT` and select one to four AFM/SEM/TEM images (`.jpg`, `.jpeg`, `.png`, `.tif`, or `.tiff`).
2. Set `Checkpoint (.pth)` to the SAM weights file.
3. Choose the SAM `Model type` (`vit_b`, `vit_l`, or `vit_h`).
4. Set `Scale (um/px)` manually, or use `autocalibration` and draw a rectangle around the scale bar.
5. Set `Max side px (AI)`. The default is `1280`; values below `256` are rejected.
6. Keep `Fast mode (CPU)` enabled for faster CPU runs, or disable it for denser SAM sampling.
7. Click `RUN AI SEGMENTATION`.
8. Select particles using click, rectangle ROI, or circle ROI tools.
9. Review measurements in `DATA STREAM`; use `Show more data` for circularity, eccentricity, and Feret diameter.
10. Use the different statistical analysis tools to study the segmented particles.
11. Export tables through `Save Data` or the save controls in each analysis window.

The GUI keeps the standard Matplotlib navigation toolbar available for pan and zoom.

## Outputs

Depending on the active tool, NanoSegment-Microscopy can save:

- Selected particle measurements as CSV or XLSX.
- Basic statistics, GMM, DBSCAN, nearest-neighbor, t-test, and IoU/Bland-Altman data as CSV or XLSX.
- Histograms, nearest-neighbor plots, false-color maps, violin/box plots, bivariate plots, and other figures as image files.
- Imported or composed images as PNG, JPEG, or TIFF.

## Troubleshooting

### SAM missing or Missing dependencies for SAM

Install PyTorch and Segment Anything:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
```

Then reinstall the project dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

### SAM checkpoint not found

Check that the `.pth` file exists and that the GUI checkpoint field points to it. If the file is in the repository root, the default `sam_vit_b_01ec64.pth` name is used automatically when launching from that directory.

### Model type must be vit_b, vit_l, or vit_h

Use one of the three SAM model types supported by the code. Make sure the model type matches the checkpoint architecture.

### Slow segmentation on CPU

Use `vit_b`, keep `Fast mode (CPU)` enabled, and reduce `Max side px (AI)`. The code resizes the input for SAM when the longest side exceeds the selected maximum.

### Scale must be a float value in um/px

Enter a positive decimal value such as `0.01`. If using a scale bar, run `autocalibration`, draw a rectangle around the scale bar, and enter the scale-bar value in nanometers when prompted.

### OpenCV is required

Autocalibration and some image-processing functions require OpenCV:

```bash
python -m pip install opencv-python
```

### Saving as .xlsx requires openpyxl

Install `openpyxl`:

```bash
python -m pip install openpyxl
```

### Tkinter window does not open

Confirm that Python has Tk support:

```bash
python -m tkinter
```

If this fails on Linux, install the system Tk package, then recreate the virtual environment if necessary.

## Citation and Acknowledgements

If NanoSegment-Microscopy is used in research, cite the original research paper:
[]

## License

This repository is distributed under the MIT License. See `LICENSE`.
