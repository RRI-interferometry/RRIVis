from astropy.io import fits
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Inspect a FITS file")
    parser.add_argument(
        "--fits-file",
        required=True,
        help="Absolute path to a FITS file to inspect",
    )
    args = parser.parse_args()

    file_path = os.path.abspath(os.path.expanduser(args.fits_file))
    print("Attempting to open file at:", file_path)

    if not os.path.isabs(file_path):
        raise ValueError("--fits-file must be an absolute path")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found at {file_path}")

    with fits.open(file_path) as hdul:
        # Print a summary of the file
        hdul.info()

        # Access specific HDU data (e.g., first HDU)
        data = hdul[0].data
        header = hdul[0].header

        # Print the header
        print("\nHeader:")
        print(repr(header))

        # If data exists, print its shape
        if data is not None:
            print("\nData shape:", data.shape)
            try:
                print("Data sample:", data[0])
            except Exception:
                pass


if __name__ == "__main__":
    main()
