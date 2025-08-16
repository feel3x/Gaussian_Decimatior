# Gaussian Splat Decimation

**Gaussian Splat Decimation** is a lightweight tool for decimating Gaussian Splat Models. It merges Gaussian points within a specified radius, allowing you to reduce model complexity while preserving overall structure.  

---

## Features

- Decimate Gaussian Splat Models using a configurable radius.
- Simple command-line interface (CLI) for fast processing.
- Output saved directly as a `.ply` file.  

---

## Installation

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the decimation tool from the command line:

```bash
python decimate.py --path_to_model <input_model.ply> --decimate_radius <radius> --save_path <output_model.ply>
```

### Arguments

| Argument           | Type    | Description                                                                 |
|------------------|--------|-----------------------------------------------------------------------------|
| `--path_to_model`   | string | Path to the Gaussian model file (required).                                  |
| `--decimate_radius` | float  | Radius in which Gaussian points are merged. Default: `0.01`.                |
| `--save_path`       | string | Full path (including `.ply`) to save the decimated model (required).       |

### Example

```bash
python decimate.py --path_to_model models/sample_gaussian.ply --decimate_radius 0.02 --save_path models/sample_decimated.ply
```

This will merge Gaussian points within a radius of `0.02` and save the decimated model.  

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit pull requests.  

---

## License

This project is primarily licensed under the **MIT License**.  

> **Note:** Portions of this project incorporate code from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting). All ownership rights to that software are held by **Inria** and the **Max Planck Institute for Informatics (MPII)**. Usage of those portions must comply with the original licensing terms of that project.  
