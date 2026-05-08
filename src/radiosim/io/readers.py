import h5py


def read_hdf5_file(file_path):
    """
    Reads the entire HDF5 file, including datasets, raw data, and attributes.

    Args:
        file_path (str): Path to the HDF5 file.
    """
    try:
        with h5py.File(file_path, "r") as h5file:
            print("Reading the entire HDF5 file...\n")
            print("File Contents:")
            print_attributes(h5file, "Root")  # Print attributes of the root group
            print_contents(h5file)
    except Exception as e:
        print(f"Error reading file: {e}")


def print_attributes(item, name):
    """
    Prints the attributes of an HDF5 group or dataset.

    Args:
        item (h5py.Group or h5py.Dataset): The HDF5 item (group or dataset).
        name (str): Name of the item being processed.
    """
    if item.attrs:
        print(f"Attributes of {name}:")
        for attr, value in item.attrs.items():
            print(f"  {attr}: {value}")


def print_contents(group, indent=0):
    """
    Recursively prints the contents of an HDF5 group, including raw data and attributes.

    Args:
        group (h5py.Group or h5py.File): The HDF5 group or file to print.
        indent (int): The current indentation level for nested groups.
    """
    for key in group:
        item = group[key]
        item_name = f"{'  ' * indent}{key}"
        if isinstance(item, h5py.Group):
            print(f"{item_name}: <Group>")
            print_attributes(item, key)  # Print attributes of the group
            print_contents(item, indent + 1)  # Recursively process the group
        elif isinstance(item, h5py.Dataset):
            print(f"{item_name}: <Dataset>")
            print_attributes(item, key)  # Print attributes of the dataset
            data = item[...]  # Get raw data
            print(f"{'  ' * (indent + 1)}Data: {data}")


if __name__ == "__main__":
    # Specify the path to the HDF5 file
    file_path = "visibility.h5"  # Update with the actual path to your file

    # Call the function to read and display the file
    read_hdf5_file(file_path)
