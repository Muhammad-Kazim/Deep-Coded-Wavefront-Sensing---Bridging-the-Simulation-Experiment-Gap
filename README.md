# **Coded_WFS_SIM**

## **Description**
`coded_wfs_sim` is a Python library for simulating light propagation through 3D structures using the Beam Propagation Method (BPM). It allows users to define structures like cubes, spheres, and planes with varying refractive indices and simulate light as it propagates through these geometries in three-dimensional space.

---

## **Features**
- **Customizable 3D Structures**: Define refractive index distributions for cubes, spheres, and other geometries.
- **Beam Propagation Method (BPM)**: Accurately model light propagation through inhomogeneous media.
- **Flexible Resolution**: Adjust spatial resolution to suit your simulation needs.
- **Extensible Framework**: Easily integrate with other tools or extend for custom simulations.

---

## **Installation**
```bash
git clone https://github.com/Muhammad-Kazim/coded_wfs_sim.git
cd coded_wfs_sim
conda env create -n custom_env_name -f environment.yml
conda activate custom_env_name
pip install -e .
```

## **Testing**
```bash
conda activate custom_env_name
python path/to/coded_wfs_sim/examples/basic_usage.py
```