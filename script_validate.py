import sys

from parser import BlueprintParser
from DSPBlueprintValidator import DSPBlueprintValidator


def main():
    if len(sys.argv) < 2:
        print("Usage: python script_validate.py <blueprint_file_or_string>")
        print("  Pass a file path or a blueprint string directly.")
        sys.exit(1)

    arg = sys.argv[1]

    # Try to read as file first, fall back to treating as raw string
    try:
        with open(arg) as f:
            bp_string = f.read().strip()
    except (FileNotFoundError, OSError):
        bp_string = arg.strip()

    bp = BlueprintParser.parse(bp_string)

    print(BlueprintParser.summary(bp))
    print()

    poly = BlueprintParser.to_polyhedron(bp)
    print(f"Polyhedron: {len(poly.vertices)}v {len(poly.edges)}e {len(poly.faces)}f")
    print()

    validation = DSPBlueprintValidator.validate_all(poly)
    print("=== Validation Results ===")
    for check, passed in validation.items():
        if check != 'all_valid':
            status = "PASS" if passed else "FAIL"
            print(f"  {check}: {status}")
    print(f"  Overall: {'ALL PASSED' if validation['all_valid'] else 'FAILED'}")

    sys.exit(0 if validation['all_valid'] else 1)


if __name__ == "__main__":
    main()
