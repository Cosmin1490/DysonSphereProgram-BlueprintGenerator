# Dyson Sphere Program Blueprint Generator

This repository contains a Python script to generate Sphere blueprints for Dyson Sphere Program. The script implements several operators from Conway Polyhedral Notation to create customizable polyhedra and outputs blueprints in a format that the game can understand. Example outputs are provided as `240.txt`, `320.txt`, and `1280.txt`.

## Dependencies

- Python 3.x

## Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Cosmin1490/DysonSphereProgram-BlueprintGenerator.git
```

2. Change into the newly created directory:

```bash
cd DysonSphereProgram-BlueprintGenerator/
```

3. Install the required packages with pip:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:

```bash
python3 script.py
```

The script will print out a blueprint.

### Customizing the blueprint

You can customize the generated blueprint by modifying the parameters in the `script.py` file. The relevant section of the code is as follows:

```python
icosahedron = Polyhedron.create_icosahedron()
icosahedron.coxeter_operator()
icosahedron.coxeter_operator()
icosahedron.dual_operator()
```

This code generates an icosahedron and applies Conway Polyhedral Notation operations to create a new polyhedron. The result can be visualized [here](https://levskaya.github.io/polyhedronisme/?recipe=A10duuI).

To change the base polyhedron, replace `Polyhedron.create_icosahedron()` with the appropriate method for the desired polyhedron.

To apply a different Conway operation, call the corresponding method on the polyhedron object.

For example, to create a dodecahedron:

```python
polyhedron = Polyhedron.create_icosahedron()
polyhedron.dual_operator()
```

**Additional Changes:**

`Polyhedron` class supports the obj format exported from [Polyhedronisme](https://levskaya.github.io/polyhedronisme/?recipe=A10duuI). This functionality can be used like so:

```python
with open(filepath, "r") as file:
    file_content = file.read()

polyhedron = Polyhedron.create_from_polyhedronisme_obj_file(file_content)
```

## Structure

The main entry point of the project is `script.py`.

- `BinaryWriter`: A class for writing binary data
- `DysonFrame`: A class for representing a frame in the blueprint
- `DysonNode`: A class for representing a node in the blueprint
- `DysonShell`: A class for representing a shell in the blueprint
- `DysonSphereLayer`: A class for representing a layer in the blueprint
- `Polyhedron`: A class for creating and manipulating polyhedra

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. I welcome any improvements or bug fixes. Please read [Contributing Guidelines](CONTRIBUTING.md) for information on how to report bugs, request features, submit pull requests, and more

## License

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more information.

